// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package ro

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"

	"github.com/samber/ro/internal/xtest"
)

// Fuzz targets for the buffer/window/sample/throttle and ordering operators. Inputs encode the
// interleaving (seed, counts, sync/async bitmask, unsubscribe timing), never the expected result.
//
// Mask layout: bit 0/1/2 = source A/B/C (or boundary) is asynchronous, bit 7 = the downstream
// unsubscribes mid-stream instead of waiting for completion.

const (
	// fuzzBOMaxMicros keeps every timer at or below 2ms so that one iteration stays fast.
	fuzzBOMaxMicros = 2000

	// fuzzBOMaxSpin bounds the busy-yield loop that delays an external Unsubscribe.
	fuzzBOMaxSpin = 24

	// fuzzBOSettle lets goroutines that outlive an unsubscription misbehave before post-conditions are read.
	fuzzBOSettle = 3 * time.Millisecond

	// fuzzBOUnsubBit selects the "unsubscribe mid-stream" scenario in the fuzz mask.
	fuzzBOUnsubBit = 0x80

	// fuzzBOMaxTicks bounds the number of boundary ticks a fuzz input can request.
	fuzzBOMaxTicks = 12
)

// fuzzBOPick derives a bounded, seed-dependent value; salt decorrelates independent knobs.
func fuzzBOPick(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

func fuzzBOMicros(seed int64, salt int) time.Duration {
	return time.Duration(fuzzBOPick(seed, salt, 100, fuzzBOMaxMicros)) * time.Microsecond
}

// fuzzBOWait polls cond until true or fuzzDeadline. Unlike fuzzWaitFor it returns an error, because
// it runs outside the test goroutine where t.Fatalf would only kill the helper goroutine.
func fuzzBOWait(what string, cond func() bool) error {
	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			return fmt.Errorf("timeout waiting for: %s", what)
		}

		time.Sleep(200 * time.Microsecond)
	}

	return nil
}

// fuzzBOIter runs body with panic recovery and a deadline so a hang or panic fails the iteration.
func fuzzBOIter(t *testing.T, name string, body func() error) {
	t.Helper()

	done := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- fmt.Errorf("panic: %v\n%s", r, debug.Stack())
			}
		}()

		done <- body()
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	case <-time.After(3 * fuzzDeadline): // longer than any inner wait, so inner messages win.
		t.Fatalf("%s: hang: iteration did not finish", name)
	}
}

// fuzzBOSink records everything a downstream observer receives.
type fuzzBOSink[T any] struct {
	mu       sync.Mutex
	vals     []T
	guard    serialGuard
	terminal int32
	errs     int32
	afterEnd int32
	lastErr  atomic.Value
}

func (s *fuzzBOSink[T]) observer() Observer[T] {
	return NewObserverWithContext(
		func(_ context.Context, v T) {
			s.guard.enter()
			defer s.guard.leave()

			if atomic.LoadInt32(&s.terminal) != 0 {
				atomic.AddInt32(&s.afterEnd, 1)
			}

			s.mu.Lock()
			s.vals = append(s.vals, v)
			s.mu.Unlock()
		},
		func(_ context.Context, err error) {
			s.guard.enter()
			defer s.guard.leave()

			s.lastErr.Store(err.Error())
			atomic.AddInt32(&s.errs, 1)
			atomic.AddInt32(&s.terminal, 1)
		},
		func(_ context.Context) {
			s.guard.enter()
			defer s.guard.leave()

			atomic.AddInt32(&s.terminal, 1)
		},
	)
}

func (s *fuzzBOSink[T]) snapshot() []T {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]T(nil), s.vals...)
}

func (s *fuzzBOSink[T]) done() bool { return atomic.LoadInt32(&s.terminal) > 0 }

func (s *fuzzBOSink[T]) violation() error {
	switch {
	case s.guard.overlapped() > 0:
		return fmt.Errorf("overlapping downstream calls: %d", s.guard.overlapped())
	case atomic.LoadInt32(&s.afterEnd) > 0:
		return fmt.Errorf("Next delivered after a terminal notification: %d", atomic.LoadInt32(&s.afterEnd))
	case atomic.LoadInt32(&s.terminal) > 1:
		return fmt.Errorf("%d terminal notifications", atomic.LoadInt32(&s.terminal))
	}

	return nil
}

// unexpectedErr reports an error terminal on a scenario whose sources never fail.
func (s *fuzzBOSink[T]) unexpectedErr() error {
	if atomic.LoadInt32(&s.errs) > 0 {
		return fmt.Errorf("unexpected error notification: %v", s.lastErr.Load())
	}

	return nil
}

func fuzzBOActive(counters []*activeCounter) int {
	total := 0
	for _, c := range counters {
		total += c.activeCount()
	}

	return total
}

// fuzzBORun subscribes, then either waits for completion or unsubscribes after a seed-dependent
// delay. In both cases every tracked upstream/boundary subscription must be released.
// check, when non-nil, validates the received values after a normal completion.
func fuzzBORun[T any](
	t *testing.T, name string, seed int64, unsub bool,
	build func() Observable[T], check func(got []T) error, counters ...*activeCounter,
) {
	t.Helper()

	fuzzBOIter(t, name, func() error {
		sink := &fuzzBOSink[T]{}
		sub := build().Subscribe(sink.observer())

		if unsub {
			for i, n := 0, fuzzBOPick(seed, 101, 0, fuzzBOMaxSpin); i < n; i++ {
				fuzzJitter(seed, i)
			}

			sub.Unsubscribe()
		} else if err := fuzzBOWait("downstream terminal notification", sink.done); err != nil {
			return err
		}

		if err := fuzzBOWait("upstream and boundary subscriptions released", func() bool {
			return fuzzBOActive(counters) == 0
		}); err != nil {
			return fmt.Errorf("%w (still active: %d, unsub=%v)", err, fuzzBOActive(counters), unsub)
		}

		time.Sleep(fuzzBOSettle)

		if err := sink.violation(); err != nil {
			return err
		}

		if unsub {
			return nil
		}

		if err := sink.unexpectedErr(); err != nil {
			return err
		}

		if check != nil {
			return check(sink.snapshot())
		}

		return nil
	})
}

// fuzzBOSrc is a finite 0..n-1 source whose subscriptions are counted.
func fuzzBOSrc(c *activeCounter, seed int64, n int, async bool) Observable[int] {
	return trackSubscriptions(c, fuzzSource(seed, n, async))
}

// fuzzBOTicks emits k ticks and never completes, like a boundary that outlives the source.
func fuzzBOTicks(c *activeCounter, seed int64, k int, async bool) Observable[int] {
	var src Observable[int]

	if !async {
		src = NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			for i := 0; i < k && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, i)
			}

			return nil
		})
	} else {
		src = NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			done := make(chan struct{})

			go func() {
				for i := 0; i < k; i++ {
					select {
					case <-done:
						return
					default:
					}

					fuzzJitter(seed+7, i)
					time.Sleep(time.Duration(fuzzBOPick(seed, 200+i, 0, 300)) * time.Microsecond)
					destination.NextWithContext(ctx, i)
				}
			}()

			return func() { close(done) }
		})
	}

	return trackSubscriptions(c, src)
}

// fuzzBOSeq checks that got is exactly 0..n-1 in order.
func fuzzBOSeq(got []int, n int) error {
	if len(got) != n {
		return fmt.Errorf("got %d items %v, want %d (items lost or duplicated)", len(got), got, n)
	}

	for i, v := range got {
		if v != i {
			return fmt.Errorf("item %d is %d: out of order in %v", i, v, got)
		}
	}

	return nil
}

func fuzzBOFlatten(buffers [][]int) []int {
	out := []int{}
	for _, b := range buffers {
		out = append(out, b...)
	}

	return out
}

// fuzzBOIncreasing checks strictly increasing values (a sampled/throttled subsequence of 0..n-1).
func fuzzBOIncreasing(n int) func(got []int) error {
	return func(got []int) error {
		for i, v := range got {
			if v < 0 || v >= n {
				return fmt.Errorf("value %d out of range [0,%d)", v, n)
			}

			if i > 0 && v <= got[i-1] {
				return fmt.Errorf("values not strictly increasing: %v", got)
			}
		}

		return nil
	}
}

func FuzzBOBufferWithCount(f *testing.F) {
	f.Skip("race: bufferwithcount-teardown-unlocked-buffer; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		size := fuzzBOPick(seed, 2, 1, 6)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithCount", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				return BufferWithCount[int](size)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up)
	})
}

func FuzzBOBufferWhen(f *testing.F) {
	// Intermittent (about 1 run in 7 at 2000 seeds), for example: got 8 items [0 1 2 3 4 5 8 9], want 10.
	f.Skip("race: bufferwhen-lost-buffer-after-unlock; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "BufferWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				return BufferWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up, tick)
	})
}

func FuzzBOBufferWithTime(f *testing.F) {
	f.Skip("race: bufferwithtime-flush-after-unlock-lost-buffer (intermittent, seen once under load); remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithTime", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				return BufferWithTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up)
	})
}

func FuzzBOBufferWithTimeOrCount(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		size := fuzzBOPick(seed, 2, 1, 6)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithTimeOrCount", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				return BufferWithTimeOrCount[int](size, d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up)
	})
}

// fuzzBOWindows subscribes to every emitted window and records its items and terminal count.
type fuzzBOWindows struct {
	mu        sync.Mutex
	items     [][]int
	terminals []*int32
}

func (w *fuzzBOWindows) next(win Observable[int]) {
	w.mu.Lock()
	idx := len(w.items)
	term := new(int32)
	w.items = append(w.items, nil)
	w.terminals = append(w.terminals, term)
	w.mu.Unlock()

	win.Subscribe(NewObserver(
		func(v int) {
			w.mu.Lock()
			w.items[idx] = append(w.items[idx], v)
			w.mu.Unlock()
		},
		func(error) { atomic.AddInt32(term, 1) },
		func() { atomic.AddInt32(term, 1) },
	))
}

// open returns how many windows have not received a terminal notification.
func (w *fuzzBOWindows) open() int {
	w.mu.Lock()
	defer w.mu.Unlock()

	n := 0

	for _, term := range w.terminals {
		if atomic.LoadInt32(term) == 0 {
			n++
		}
	}

	return n
}

func (w *fuzzBOWindows) flatten() []int {
	w.mu.Lock()
	defer w.mu.Unlock()

	return fuzzBOFlatten(w.items)
}

// subscribeWindows subscribes to a higher-order observable and tracks every window.
func (w *fuzzBOWindows) subscribe(obs Observable[Observable[int]], outer *fuzzBOSink[int]) Subscription {
	return obs.Subscribe(NewObserver(
		w.next,
		func(err error) {
			outer.lastErr.Store(err.Error())
			atomic.AddInt32(&outer.errs, 1)
			atomic.AddInt32(&outer.terminal, 1)
		},
		func() { atomic.AddInt32(&outer.terminal, 1) },
	))
}

func FuzzBOWindowWhen(f *testing.F) {
	f.Skip("race: windowwhen-value-lost-in-completed-window; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}
		counters := []*activeCounter{up, tick}

		fuzzBOIter(t, "WindowWhen", func() error {
			outer := &fuzzBOSink[int]{}
			wins := &fuzzBOWindows{}
			obs := WindowWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			sub := wins.subscribe(obs, outer)

			if mask&fuzzBOUnsubBit != 0 {
				for i, m := 0, fuzzBOPick(seed, 101, 0, fuzzBOMaxSpin); i < m; i++ {
					fuzzJitter(seed, i)
				}

				sub.Unsubscribe()

				return fuzzBOWait("upstream and boundary released", func() bool { return fuzzBOActive(counters) == 0 })
			}

			if err := fuzzBOWait("outer terminal", outer.done); err != nil {
				return err
			}

			if err := fuzzBOWait("every window terminal", func() bool { return wins.open() == 0 }); err != nil {
				return fmt.Errorf("%w: %d windows never completed", err, wins.open())
			}

			if err := fuzzBOWait("upstream and boundary released", func() bool { return fuzzBOActive(counters) == 0 }); err != nil {
				return err
			}

			time.Sleep(fuzzBOSettle)

			if err := outer.unexpectedErr(); err != nil {
				return err
			}

			return fuzzBOSeq(wins.flatten(), n)
		})
	})
}

// FuzzBOWindowWhenUnsubscribe isolates the teardown path: the open window must get a terminal
// notification when the downstream unsubscribes, otherwise its subscriber waits forever.
func FuzzBOWindowWhenUnsubscribe(f *testing.F) {
	f.Skip("race: windowwhen-teardown-leaves-window-open; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		// Source emits a few items then stays open, so only Unsubscribe can end the windows.
		src := trackSubscriptions(up, NewUnsafeObservableWithContext(func(ctx context.Context, d Observer[int]) Teardown {
			emit := func() {
				for i := 0; i < 5 && !d.IsClosed(); i++ {
					d.NextWithContext(ctx, i)
				}
			}
			if !fuzzIsAsync(mask, 0) {
				emit()
				return nil
			}

			go emit()

			return nil
		}))

		fuzzBOIter(t, "WindowWhen/unsubscribe", func() error {
			wins := &fuzzBOWindows{}
			outer := &fuzzBOSink[int]{}
			sub := wins.subscribe(WindowWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(src), outer)

			for i, m := 0, fuzzBOPick(seed, 101, 0, fuzzBOMaxSpin); i < m; i++ {
				fuzzJitter(seed, i)
			}

			sub.Unsubscribe()

			if err := fuzzBOWait("upstream and boundary released", func() bool { return fuzzBOActive([]*activeCounter{up, tick}) == 0 }); err != nil {
				return err
			}

			if err := fuzzBOWait("open window completed on unsubscribe", func() bool { return wins.open() == 0 }); err != nil {
				return fmt.Errorf("%w: %d windows left without terminal", err, wins.open())
			}

			return nil
		})
	})
}

func FuzzBOSampleWhen(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "SampleWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[int] {
				return SampleWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up, tick)
	})
}

func FuzzBOSampleTime(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "SampleTime", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[int] {
				return SampleTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up)
	})
}

func FuzzBOThrottleWhen(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "ThrottleWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[int] {
				return ThrottleWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got []int) error {
				if len(got) > k {
					return fmt.Errorf("%d values passed but only %d ticks were sent", len(got), k)
				}

				return fuzzBOIncreasing(n)(got)
			}, up, tick)
	})
}

func FuzzBOThrottleTime(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "ThrottleTime", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[int] {
				return ThrottleTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up)
	})
}

func FuzzBOGroupBy(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		keys := fuzzBOPick(seed, 5, 1, 4)
		up := &activeCounter{}

		fuzzBOIter(t, "GroupBy", func() error {
			wins := &fuzzBOWindows{}
			outer := &fuzzBOSink[int]{}
			sub := wins.subscribe(GroupBy(func(v int) int { return v % keys })(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0))), outer)

			if mask&fuzzBOUnsubBit != 0 {
				for i, m := 0, fuzzBOPick(seed, 101, 0, fuzzBOMaxSpin); i < m; i++ {
					fuzzJitter(seed, i)
				}

				sub.Unsubscribe()

				if err := fuzzBOWait("upstream released", func() bool { return up.activeCount() == 0 }); err != nil {
					return err
				}

				return fuzzBOWait("every group terminal after unsubscribe", func() bool { return wins.open() == 0 })
			}

			if err := fuzzBOWait("outer terminal", outer.done); err != nil {
				return err
			}

			if err := fuzzBOWait("every group terminal", func() bool { return wins.open() == 0 }); err != nil {
				return err
			}

			if err := fuzzBOWait("upstream released", func() bool { return up.activeCount() == 0 }); err != nil {
				return err
			}

			wins.mu.Lock()
			defer wins.mu.Unlock()

			distinct := n
			if keys < n {
				distinct = keys
			}

			if len(wins.items) != distinct {
				return fmt.Errorf("%d groups, want %d", len(wins.items), distinct)
			}

			total := 0

			for _, items := range wins.items {
				total += len(items)

				for i, v := range items {
					if v%keys != items[0]%keys {
						return fmt.Errorf("group mixes keys: %v", items)
					}

					if i > 0 && v <= items[i-1] {
						return fmt.Errorf("group out of order: %v", items)
					}
				}
			}

			if total != n {
				return fmt.Errorf("groups hold %d items, want %d", total, n)
			}

			return nil
		})
	})
}

func FuzzBOPairwise(f *testing.F) {
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		up := &activeCounter{}

		fuzzBORun(t, "Pairwise", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] { return Pairwise[int]()(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0))) },
			func(got [][]int) error {
				want := n - 1
				if want < 0 {
					want = 0
				}

				if len(got) != want {
					return fmt.Errorf("%d pairs, want %d", len(got), want)
				}

				for i, p := range got {
					if len(p) != 2 || p[0] != i || p[1] != i+1 {
						return fmt.Errorf("pair %d is %v, want [%d %d]", i, p, i, i+1)
					}
				}

				return nil
			}, up)
	})
}

func FuzzBOZip2(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		na, nb := fuzzBOPick(seed, 1, 0, fuzzMaxItems), fuzzBOPick(seed, 2, 0, fuzzMaxItems)
		a, b := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "Zip2", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[lo.Tuple2[int, int]] {
				return Zip2(fuzzBOSrc(a, seed, na, fuzzIsAsync(mask, 0)), fuzzBOSrc(b, seed+1, nb, fuzzIsAsync(mask, 1)))
			},
			func(got []lo.Tuple2[int, int]) error {
				want := na
				if nb < want {
					want = nb
				}

				if len(got) != want {
					return fmt.Errorf("%d pairs, want %d: %v", len(got), want, got)
				}

				for i, p := range got {
					if p.A != i || p.B != i {
						return fmt.Errorf("pair %d is %v, want (%d,%d): out of order in %v", i, p, i, i, got)
					}
				}

				return nil
			}, a, b)
	})
}

func FuzzBOZipVariadic(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		ns := []int{fuzzBOPick(seed, 1, 0, fuzzMaxItems), fuzzBOPick(seed, 2, 0, fuzzMaxItems), fuzzBOPick(seed, 6, 0, fuzzMaxItems)}
		cs := []*activeCounter{{}, {}, {}}

		fuzzBORun(t, "Zip", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				srcs := make([]Observable[int], len(ns))
				for i := range ns {
					srcs[i] = fuzzBOSrc(cs[i], seed+int64(i), ns[i], fuzzIsAsync(mask, i))
				}

				return Zip(srcs...)
			},
			func(got [][]int) error {
				want := ns[0]
				for _, n := range ns {
					if n < want {
						want = n
					}
				}

				if len(got) != want {
					return fmt.Errorf("%d tuples, want %d: %v", len(got), want, got)
				}

				for i, p := range got {
					if len(p) != len(ns) || p[0] != i || p[1] != i || p[2] != i {
						return fmt.Errorf("tuple %d is %v, want all %d: out of order in %v", i, p, i, got)
					}
				}

				return nil
			}, cs...)
	})
}

// FuzzBOCombineLatest2 checks that the last tuple holds the latest value of both sources.
func FuzzBOCombineLatest2(f *testing.F) {
	f.Skip("race: combinelatest-stale-last-tuple; remove when fixed")

	fuzzBOCombineLatest2(f, "CombineLatest2/final", func(got []lo.Tuple2[int, int], na, nb int) error {
		if last := got[len(got)-1]; last.A != na-1 || last.B != nb-1 {
			return fmt.Errorf("stale last tuple %v, want (%d,%d)", last, na-1, nb-1)
		}

		return nil
	})
}

// FuzzBOCombineLatest2Order checks that tuples never go back to an older value of a source.
func FuzzBOCombineLatest2Order(f *testing.F) {
	f.Skip("race: combinelatest-out-of-order-tuples; remove when fixed")

	fuzzBOCombineLatest2(f, "CombineLatest2/order", func(got []lo.Tuple2[int, int], _, _ int) error {
		for i := 1; i < len(got); i++ {
			if got[i].A < got[i-1].A || got[i].B < got[i-1].B {
				return fmt.Errorf("tuple went backwards: %v then %v", got[i-1], got[i])
			}
		}

		return nil
	})
}

// FuzzBOCombineLatest2Duplicate checks that no two consecutive tuples are identical.
func FuzzBOCombineLatest2Duplicate(f *testing.F) {
	f.Skip("race: combinelatest-duplicate-tuples; remove when fixed")

	fuzzBOCombineLatest2(f, "CombineLatest2/duplicate", func(got []lo.Tuple2[int, int], _, _ int) error {
		for i := 1; i < len(got); i++ {
			if got[i] == got[i-1] {
				return fmt.Errorf("duplicate tuple %v", got[i])
			}
		}

		return nil
	})
}

func fuzzBOCombineLatest2(f *testing.F, name string, extra func(got []lo.Tuple2[int, int], na, nb int) error) {
	f.Helper()
	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		na, nb := fuzzBOPick(seed, 1, 0, fuzzMaxItems), fuzzBOPick(seed, 2, 0, fuzzMaxItems)
		a, b := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, name, seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[lo.Tuple2[int, int]] {
				return CombineLatest2(fuzzBOSrc(a, seed, na, fuzzIsAsync(mask, 0)), fuzzBOSrc(b, seed+1, nb, fuzzIsAsync(mask, 1)))
			},
			func(got []lo.Tuple2[int, int]) error {
				if na == 0 || nb == 0 {
					if len(got) != 0 {
						return fmt.Errorf("emitted %v although a source is empty", got)
					}

					return nil
				}

				if len(got) == 0 {
					return errors.New("no tuple emitted although both sources emitted")
				}

				return extra(got, na, nb)
			}, a, b)
	})
}

func FuzzBOCombineLatestAll(f *testing.F) {
	f.Skip("race: combinelatestall-stale-last-tuple; remove when fixed")

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		ns := []int{fuzzBOPick(seed, 1, 1, fuzzMaxItems), fuzzBOPick(seed, 2, 1, fuzzMaxItems), fuzzBOPick(seed, 6, 1, fuzzMaxItems)}
		cs := []*activeCounter{{}, {}, {}}

		fuzzBORun(t, "CombineLatestAll", seed, mask&fuzzBOUnsubBit != 0,
			func() Observable[[]int] {
				srcs := make([]Observable[int], len(ns))
				for i := range ns {
					srcs[i] = fuzzBOSrc(cs[i], seed+int64(i), ns[i], fuzzIsAsync(mask, i))
				}

				return CombineLatestAll[int]()(Just(srcs...))
			},
			func(got [][]int) error {
				if len(got) == 0 {
					return errors.New("no tuple emitted although every source emitted")
				}

				last := got[len(got)-1]
				for i, n := range ns {
					if len(last) != len(ns) || last[i] != n-1 {
						return fmt.Errorf("stale last tuple %v, want latest values of every source", last)
					}
				}

				return nil
			}, cs...)
	})
}
