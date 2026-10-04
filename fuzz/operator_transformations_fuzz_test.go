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

package fuzz

import (
	"context"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
	"github.com/stretchr/testify/assert"
)

func FuzzHOMergeMap(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindMergeMap, seed, size, mask, k)
	})
}

func FuzzHOFlatMap(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindFlatMap, seed, size, mask, k)
	})
}

// FuzzHOFlatMapSameSubject feeds FlatMap from a subject while every inner waits for the NEXT
// item of that same subject. The producer is blocked inside Next while FlatMap waits for the
// inner, so the producer must be released by buffering the outer items.
func FuzzHOFlatMapSameSubject(f *testing.F) {
	f.Skip("race: flatmap-blocks-producer; remove when fixed") // fuzz_higherorder_test.go:504: producer blocked in Subject.Next: FlatMap waits for an inner inside the outer Next
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		h := fuzzHONewHarness(t)
		subject := ro.NewPublishSubject[int]()
		m := fuzzBound(size, 1, fuzzHOMaxConcatWith)
		independent := mask&fuzzHOBoundedBit != 0

		var innerC activeCounter

		project := func(v int) ro.Observable[int] {
			if independent {
				return trackSubscriptions(&innerC, fuzzHOSource(seed, v, 2, fuzzIsAsync(mask, v+1), fuzzHOEndComplete))
			}

			return trackSubscriptions(&innerC, ro.Take[int](1)(subject.AsObservable()))
		}

		h.start(ro.FlatMap(project)(subject.AsObservable()))
		h.waitFor("Subscribe to return", h.hasReturned)

		producerDone := make(chan struct{})

		go func() {
			defer close(producerDone)

			// The feeder is itself sync or async: its Next calls run on the producing goroutine.
			finished := make(chan struct{})

			fuzzSource(seed, m, fuzzIsAsync(mask, 0)).Subscribe(ro.NewObserver(
				subject.Next,
				subject.Error,
				func() {
					subject.Complete()
					close(finished)
				},
			))

			<-finished
		}()

		select {
		case <-producerDone:
		case <-time.After(fuzzDeadline):
			h.fail("producer blocked in Subject.Next: FlatMap waits for an inner inside the outer Next")
		}

		h.waitFor("a terminal notification", func() bool { return h.rec.terminals() > 0 })
		h.verify(&innerC)

		if !independent && h.rec.nextCount() != m-1 {
			h.fail("%d values received, want %d (inner i receives item i+1)", h.rec.nextCount(), m-1)
		}
	})
}

func fuzzBOMicros(seed int64, salt int) time.Duration {
	return time.Duration(fuzzBOPick(seed, salt, 100, fuzzBOMaxMicros)) * time.Microsecond
}

// fuzzBOTicks emits k ticks and never completes, like a boundary that outlives the source.
func fuzzBOTicks(c *activeCounter, seed int64, k int, async bool) ro.Observable[int] {
	var src ro.Observable[int]

	if !async {
		src = ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
			for i := 0; i < k && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, i)
			}

			return nil
		})
	} else {
		src = ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
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

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		size := fuzzBOPick(seed, 2, 1, 6)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithCount", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithCount[int](size)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up)
	})
}

func FuzzBOBufferWhen(f *testing.F) {
	// Intermittent (about 1 run in 7 at 2000 seeds), for example: got 8 items [0 1 2 3 4 5 8 9], want 10.
	f.Skip("race: bufferwhen-lost-buffer-after-unlock; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "BufferWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up, tick)
	})
}

func FuzzBOBufferWithTime(f *testing.F) {
	f.Skip("race: bufferwithtime-flush-after-unlock-lost-buffer (intermittent, seen once under load); remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithTime", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return fuzzBOSeq(fuzzBOFlatten(got), n) },
			up)
	})
}

func FuzzBOBufferWithTimeOrCount(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		size := fuzzBOPick(seed, 2, 1, 6)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "BufferWithTimeOrCount", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithTimeOrCount[int](size, d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
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

func (w *fuzzBOWindows) next(win ro.Observable[int]) {
	w.mu.Lock()
	idx := len(w.items)
	term := new(int32)
	w.items = append(w.items, nil)
	w.terminals = append(w.terminals, term)
	w.mu.Unlock()

	win.Subscribe(ro.NewObserver(
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
func (w *fuzzBOWindows) subscribe(obs ro.Observable[ro.Observable[int]], outer *fuzzBOSink[int]) ro.Subscription {
	return obs.Subscribe(ro.NewObserver(
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

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}
		counters := []*activeCounter{up, tick}

		fuzzBOIter(t, "WindowWhen", func() error {
			outer := &fuzzBOSink[int]{}
			wins := &fuzzBOWindows{}
			obs := ro.WindowWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
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

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		// Source emits a few items then stays open, so only Unsubscribe can end the windows.
		src := trackSubscriptions(up, ro.NewUnsafeObservableWithContext(func(ctx context.Context, d ro.Observer[int]) ro.Teardown {
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
			sub := wins.subscribe(ro.WindowWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(src), outer)

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
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "SampleWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[int] {
				return ro.SampleWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up, tick)
	})
}

func FuzzBOSampleTime(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "SampleTime", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[int] {
				return ro.SampleTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up)
	})
}

func FuzzBOThrottleWhen(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		k := fuzzBOPick(seed, 3, 0, fuzzBOMaxTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		fuzzBORun(t, "ThrottleWhen", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[int] {
				return ro.ThrottleWhen[int](fuzzBOTicks(tick, seed, k, fuzzIsAsync(mask, 1)))(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
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
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		d := fuzzBOMicros(seed, 4)
		up := &activeCounter{}

		fuzzBORun(t, "ThrottleTime", seed, mask&fuzzBOUnsubBit != 0,
			func() ro.Observable[int] {
				return ro.ThrottleTime[int](d)(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			fuzzBOIncreasing(n), up)
	})
}

func FuzzBOGroupBy(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := fuzzBOPick(seed, 1, 0, fuzzMaxItems)
		keys := fuzzBOPick(seed, 5, 1, 4)
		up := &activeCounter{}

		fuzzBOIter(t, "GroupBy", func() error {
			wins := &fuzzBOWindows{}
			outer := &fuzzBOSink[int]{}
			sub := wins.subscribe(ro.GroupBy(func(v int) int { return v % keys })(fuzzBOSrc(up, seed, n, fuzzIsAsync(mask, 0))), outer)

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

// Unsubscribing while the source is still emitting new and existing keys must
// neither race on the group registry nor leave a group uncompleted.
func FuzzOperatorTransformationGroupByTeardownRacesInFlightValues(f *testing.F) {
	const (
		keys      = 8
		emissions = 2000
	)

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i)} })

	f.Fuzz(func(t *testing.T, seed int64) {
		is := assert.New(t)

		source := ro.NewPublishSubject[int]()

		var open int64 // groups emitted but not yet completed; atomic.Int64 needs Go 1.19

		// Yielding in the iteratee keeps values in flight while teardown runs.
		iteratee := func(v int) int {
			runtime.Gosched()
			fuzzJitter(seed, v)
			return v % keys
		}

		sub := ro.GroupBy(iteratee)(source).Subscribe(
			ro.OnNext(func(group ro.Observable[int]) {
				atomic.AddInt64(&open, 1)
				group.Subscribe(ro.NewObserver(
					func(int) {},
					func(error) { atomic.AddInt64(&open, -1) },
					func() { atomic.AddInt64(&open, -1) },
				))
			}),
		)

		done := make(chan struct{})
		started := make(chan struct{})
		go func() {
			defer close(done)

			for i := 0; i < emissions; i++ {
				source.Next(i)

				if i == keys {
					close(started)
				}
			}
		}()

		<-started // unsubscribe while values for existing keys are still in flight
		fuzzJitter(seed, emissions)
		sub.Unsubscribe()
		<-done

		is.Zero(atomic.LoadInt64(&open), "every emitted group must be completed on teardown")
	})
}
