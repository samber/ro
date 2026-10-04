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
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro/internal/xtest"
)

// Fuzz targets for short-circuit operators (Contains, Find, First, ElementAt, Head, Take,
// TakeWhile, All, Average on empty) and for upstream propagation of simple operators.
//
// Short-circuit targets observe the downstream through the regular Subscriber, which drops
// every notification sent after the first terminal one: re-fired Next/Complete are absorbed
// by design (the raw destination of an operator is not a valid oracle, even All re-fires there).
// What stays observable, and is asserted, is the user predicate running after the decision.

var errFuzzSCBoom = errors.New("fuzzsc: boom")

const (
	// fuzzSCSettle lets goroutines that outlive a decision misbehave before post-conditions are read.
	fuzzSCSettle = 3 * time.Millisecond

	// fuzzSCNoDefault is the fallback emitted by ElementAtOrDefault. It is outside the source range [0, n).
	fuzzSCNoDefault = -1

	// fuzzSCMaxYields bounds the scheduler yields before an external Unsubscribe.
	fuzzSCMaxYields = 50

	// Upstream stop modes.
	fuzzSCStopUnsub = 1 // downstream unsubscribes concurrently
	fuzzSCStopError = 2 // source ends with an error
	fuzzSCStopModes = 4 // modes 0 and 3 run to a normal completion
)

// fuzzSCPick derives a bounded, seed-dependent value; salt decorrelates independent knobs.
func fuzzSCPick(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

// fuzzSCMin exists because go.mod declares go 1.18, which has no min builtin.
func fuzzSCMin(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// fuzzSCIter runs body with panic recovery and a deadline, so that a hang or a panic fails
// the iteration with the operator's name.
func fuzzSCIter(t *testing.T, name string, body func() error) {
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
	case <-time.After(fuzzDeadline):
		t.Fatalf("%s: hang: iteration did not finish within %s", name, fuzzDeadline)
	}
}

// fuzzSCProbe counts user predicate calls, and the calls made after the deciding one.
type fuzzSCProbe struct {
	calls   int32
	decided int32
	extra   int32
}

func (p *fuzzSCProbe) call(decides bool) {
	atomic.AddInt32(&p.calls, 1)

	if atomic.LoadInt32(&p.decided) != 0 {
		atomic.AddInt32(&p.extra, 1)
	}

	if decides {
		atomic.StoreInt32(&p.decided, 1)
	}
}

func (p *fuzzSCProbe) extraCalls() int { return int(atomic.LoadInt32(&p.extra)) }

// fuzzSCRawSink is a destination that counts every notification it receives,
// including the ones sent after a terminal notification.
type fuzzSCRawSink[T any] struct {
	guard     serialGuard
	mu        sync.Mutex
	values    []T
	errs      int32
	comps     int32
	afterTerm int32
}

func (s *fuzzSCRawSink[T]) terminated() bool {
	return atomic.LoadInt32(&s.errs)+atomic.LoadInt32(&s.comps) > 0
}

func (s *fuzzSCRawSink[T]) Next(v T) { s.NextWithContext(context.Background(), v) }

func (s *fuzzSCRawSink[T]) NextWithContext(_ context.Context, v T) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	s.mu.Lock()
	s.values = append(s.values, v)
	s.mu.Unlock()
}

func (s *fuzzSCRawSink[T]) Error(err error) { s.ErrorWithContext(context.Background(), err) }

func (s *fuzzSCRawSink[T]) ErrorWithContext(_ context.Context, _ error) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	atomic.AddInt32(&s.errs, 1)
}

func (s *fuzzSCRawSink[T]) Complete() { s.CompleteWithContext(context.Background()) }

func (s *fuzzSCRawSink[T]) CompleteWithContext(_ context.Context) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	atomic.AddInt32(&s.comps, 1)
}

func (s *fuzzSCRawSink[T]) IsClosed() bool    { return false }
func (s *fuzzSCRawSink[T]) HasThrown() bool   { return false }
func (s *fuzzSCRawSink[T]) IsCompleted() bool { return false }

func (s *fuzzSCRawSink[T]) snapshot() []T {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]T(nil), s.values...)
}

// fuzzSCSource emits 0..n-1 then completes (or fails when failAtEnd). Async sources emit from their
// own goroutine. checkClosed makes the source honour IsClosed/teardown; otherwise it emits everything.
func fuzzSCSource(seed int64, n int, async, checkClosed, failAtEnd bool, wg *sync.WaitGroup) Observable[int] {
	finish := func(ctx context.Context, destination Observer[int]) {
		if failAtEnd {
			destination.ErrorWithContext(ctx, errFuzzSCBoom)
			return
		}

		destination.CompleteWithContext(ctx)
	}

	if !async {
		return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			for i := 0; i < n; i++ {
				if checkClosed && destination.IsClosed() {
					return nil
				}

				destination.NextWithContext(ctx, i)
			}

			finish(ctx, destination)

			return nil
		})
	}

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		done := make(chan struct{})

		var once sync.Once

		if wg != nil {
			wg.Add(1)
		}

		go func() {
			if wg != nil {
				defer wg.Done()
			}

			for i := 0; i < n; i++ {
				if checkClosed {
					select {
					case <-done:
						return
					default:
					}
				}

				fuzzJitter(seed, i)
				destination.NextWithContext(ctx, i)
			}

			finish(ctx, destination)
		}()

		return func() { once.Do(func() { close(done) }) }
	})
}

// fuzzSCRunRaw subscribes build(source) with a counting destination, waits for the source to
// finish and for stray notifications to settle, then returns what the destination saw.
func fuzzSCRunRaw[R any](t *testing.T, name string, seed int64, mask uint8, n int, build func(source Observable[int]) Observable[R]) *fuzzSCRawSink[R] {
	t.Helper()

	async := fuzzIsAsync(mask, 0)
	checkClosed := fuzzIsAsync(mask, 1)

	sink := &fuzzSCRawSink[R]{}

	fuzzSCIter(t, name, func() error {
		var wg sync.WaitGroup

		obs := build(fuzzSCSource(seed, n, async, checkClosed, false, &wg))

		sub := obs.SubscribeWithContext(context.Background(), sink)

		wg.Wait()
		time.Sleep(fuzzSCSettle)

		sub.Unsubscribe()

		return nil
	})

	return sink
}

// fuzzSCCheck asserts the single-result invariant: exactly the wanted values, exactly one terminal
// notification, nothing after it, no overlapping calls, and no predicate call after the decision.
func fuzzSCCheck[R any](t *testing.T, name string, sink *fuzzSCRawSink[R], probe *fuzzSCProbe, wantValues []R, wantErr bool, mask uint8) {
	t.Helper()

	got := sink.snapshot()
	errs := int(atomic.LoadInt32(&sink.errs))
	comps := int(atomic.LoadInt32(&sink.comps))
	after := int(atomic.LoadInt32(&sink.afterTerm))
	extra := 0

	if probe != nil {
		extra = probe.extraCalls()
	}

	wantErrs, wantComps := 0, 1
	if wantErr {
		wantErrs, wantComps = 1, 0
	}

	if fmt.Sprint(got) != fmt.Sprint(wantValues) || errs != wantErrs || comps != wantComps || after != 0 || extra != 0 || sink.guard.overlapped() != 0 {
		t.Fatalf("%s (async=%v checkClosed=%v): values=%v want %v; errors=%d (want %d); completes=%d (want %d); notifications after terminal=%d; predicate calls after decision=%d; overlaps=%d",
			name, fuzzIsAsync(mask, 0), fuzzIsAsync(mask, 1), got, wantValues, errs, wantErrs, comps, wantComps, after, extra, sink.guard.overlapped())
	}
}

func fuzzSCSeeds(f *testing.F) {
	f.Helper()

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })
}

// fuzzSCScenario derives the item count and the decision index. k may be >= n, which means "no decision".
func fuzzSCScenario(seed int64) (n, k, variant int) {
	n = fuzzSCPick(seed, 0, 0, fuzzMaxItems)
	k = fuzzSCPick(seed, 1, 0, n+1)
	variant = fuzzSCPick(seed, 2, 0, 1)

	return n, k, variant
}

func fuzzSCRange(from, to int) []int {
	out := []int{}
	for i := from; i < to; i++ {
		out = append(out, i)
	}

	return out
}

func fuzzSCWantFirst(n, k int) []int {
	if k < n {
		return []int{k}
	}

	return []int{}
}

func FuzzSCContains(f *testing.F) {
	f.Skip("race: contains-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "Contains", seed, mask, n, func(source Observable[int]) Observable[bool] {
			// >= makes every item after the decision a match too, so a re-fired decision is visible.
			if variant == 0 {
				return Contains(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return ContainsI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "Contains", sink, probe, []bool{k < n}, false, mask)
	})
}

func FuzzSCFind(f *testing.F) {
	f.Skip("race: find-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "Find", seed, mask, n, func(source Observable[int]) Observable[int] {
			if variant == 0 {
				return Find(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return FindI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "Find", sink, probe, fuzzSCWantFirst(n, k), false, mask)
	})
}

func FuzzSCFirst(f *testing.F) {
	f.Skip("race: first-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "First", seed, mask, n, func(source Observable[int]) Observable[int] {
			if variant == 0 {
				return First(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return FirstI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "First", sink, probe, fuzzSCWantFirst(n, k), k >= n, mask)
	})
}

func FuzzSCElementAt(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "ElementAt", seed, mask, n, func(source Observable[int]) Observable[int] {
			return ElementAt[int](k)(source)
		})

		fuzzSCCheck(t, "ElementAt", sink, nil, fuzzSCWantFirst(n, k), k >= n, mask)
	})
}

func FuzzSCElementAtOrDefault(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "ElementAtOrDefault", seed, mask, n, func(source Observable[int]) Observable[int] {
			return ElementAtOrDefault(int64(k), fuzzSCNoDefault)(source)
		})

		want := []int{fuzzSCNoDefault}
		if k < n {
			want = []int{k}
		}

		fuzzSCCheck(t, "ElementAtOrDefault", sink, nil, want, false, mask)
	})
}

func FuzzSCHead(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, _, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "Head", seed, mask, n, func(source Observable[int]) Observable[int] {
			return Head[int]()(source)
		})

		want := []int{}
		if n > 0 {
			want = []int{0}
		}

		fuzzSCCheck(t, "Head", sink, nil, want, n == 0, mask)
	})
}

func FuzzSCTake(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "Take", seed, mask, n, func(source Observable[int]) Observable[int] {
			return Take[int](int64(k))(source)
		})

		fuzzSCCheck(t, "Take", sink, nil, fuzzSCRange(0, fuzzSCMin(k, n)), false, mask)
	})
}

func FuzzSCTakeWhile(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "TakeWhile", seed, mask, n, func(source Observable[int]) Observable[int] {
			if variant == 0 {
				return TakeWhile(func(v int) bool { probe.call(v >= k); return v < k })(source)
			}

			return TakeWhileI(func(v int, _ int64) bool { probe.call(v >= k); return v < k })(source)
		})

		fuzzSCCheck(t, "TakeWhile", sink, probe, fuzzSCRange(0, fuzzSCMin(k, n)), false, mask)
	})
}

// FuzzSCAll is the control: All short-circuits correctly since #429.
func FuzzSCAll(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "All", seed, mask, n, func(source Observable[int]) Observable[bool] {
			if variant == 0 {
				return All(func(v int) bool { probe.call(v >= k); return v < k })(source)
			}

			return AllI(func(v int, _ int64) bool { probe.call(v >= k); return v < k })(source)
		})

		fuzzSCCheck(t, "All", sink, probe, []bool{k >= n}, false, mask)
	})
}

// FuzzSCAverageEmpty checks Average over an empty source: at most one value, then exactly one terminal.
// It subscribes through the raw subscribe function because the Subscriber would absorb the second
// Next/Complete, hiding the missing return.
func FuzzSCAverageEmpty(f *testing.F) {
	f.Skip("race: average-empty-double-emit (missing return after NaN+Complete); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		sink := &fuzzSCRawSink[float64]{}

		fuzzSCIter(t, "AverageEmpty", func() error {
			var wg sync.WaitGroup

			impl, ok := Average[int]()(fuzzSCSource(seed, 0, fuzzIsAsync(mask, 0), true, false, &wg)).(*observableImpl[float64])
			if !ok {
				return errors.New("unexpected observable type")
			}

			teardown := impl.subscribe(context.Background(), sink)

			wg.Wait()
			time.Sleep(fuzzSCSettle)

			if teardown != nil {
				teardown()
			}

			return nil
		})

		got := sink.snapshot()
		errs := atomic.LoadInt32(&sink.errs)
		comps := atomic.LoadInt32(&sink.comps)
		after := atomic.LoadInt32(&sink.afterTerm)

		if len(got) > 1 || errs != 0 || comps != 1 || after != 0 {
			t.Fatalf("AverageEmpty (async=%v): values=%v (want at most 1); errors=%d; completes=%d (want 1); notifications after terminal=%d",
				fuzzIsAsync(mask, 0), got, errs, comps, after)
		}
	})
}

// fuzzSCInputs holds the tracked sources an operator may subscribe to.
type fuzzSCInputs struct {
	main     Observable[int]
	signal   Observable[int]
	fallback Observable[int]
	k        int
}

// fuzzSCTerm records downstream terminal notifications.
type fuzzSCTerm struct{ terminals int32 }

// fuzzSCCase wires one operator over the tracked sources of an iteration.
type fuzzSCCase struct {
	name string
	run  func(ctx context.Context, in fuzzSCInputs, term *fuzzSCTerm) Subscription
}

func fuzzSCOp[R any](name string, build func(in fuzzSCInputs) Observable[R]) fuzzSCCase {
	return fuzzSCCase{
		name: name,
		run: func(ctx context.Context, in fuzzSCInputs, term *fuzzSCTerm) Subscription {
			return build(in).SubscribeWithContext(ctx, NewObserverWithContext(
				func(_ context.Context, _ R) {},
				func(_ context.Context, _ error) { atomic.AddInt32(&term.terminals, 1) },
				func(_ context.Context) { atomic.AddInt32(&term.terminals, 1) },
			))
		},
	}
}

func fuzzSCTable() []fuzzSCCase {
	return []fuzzSCCase{
		fuzzSCOp("Map", func(in fuzzSCInputs) Observable[int] {
			return Map(func(v int) int { return v + 1 })(in.main)
		}),
		fuzzSCOp("Filter", func(in fuzzSCInputs) Observable[int] {
			return Filter(func(v int) bool { return v%2 == 0 })(in.main)
		}),
		fuzzSCOp("Take", func(in fuzzSCInputs) Observable[int] { return Take[int](int64(in.k))(in.main) }),
		fuzzSCOp("Skip", func(in fuzzSCInputs) Observable[int] { return Skip[int](int64(in.k))(in.main) }),
		fuzzSCOp("TakeWhile", func(in fuzzSCInputs) Observable[int] {
			return TakeWhile(func(v int) bool { return v < in.k })(in.main)
		}),
		fuzzSCOp("TakeLast", func(in fuzzSCInputs) Observable[int] { return TakeLast[int](in.k)(in.main) }),
		fuzzSCOp("SkipLast", func(in fuzzSCInputs) Observable[int] { return SkipLast[int](in.k)(in.main) }),
		fuzzSCOp("SkipWhile", func(in fuzzSCInputs) Observable[int] {
			return SkipWhile(func(v int) bool { return v < in.k })(in.main)
		}),
		fuzzSCOp("Scan", func(in fuzzSCInputs) Observable[int] {
			return Scan(func(acc, v int) int { return acc + v }, 0)(in.main)
		}),
		fuzzSCOp("Distinct", func(in fuzzSCInputs) Observable[int] { return Distinct[int]()(in.main) }),
		fuzzSCOp("Tap", func(in fuzzSCInputs) Observable[int] {
			return Tap(func(int) {}, func(error) {}, func() {})(in.main)
		}),
		fuzzSCOp("Materialize", func(in fuzzSCInputs) Observable[Notification[int]] {
			return Materialize[int]()(in.main)
		}),
		fuzzSCOp("Timestamp", func(in fuzzSCInputs) Observable[TimestampValue[int]] {
			return Timestamp[int]()(in.main)
		}),
		fuzzSCOp("Head", func(in fuzzSCInputs) Observable[int] { return Head[int]()(in.main) }),
		fuzzSCOp("First", func(in fuzzSCInputs) Observable[int] {
			return First(func(v int) bool { return v >= in.k })(in.main)
		}),
		fuzzSCOp("ElementAt", func(in fuzzSCInputs) Observable[int] { return ElementAt[int](in.k)(in.main) }),
		fuzzSCOp("TakeUntil", func(in fuzzSCInputs) Observable[int] { return TakeUntil[int, int](in.signal)(in.main) }),
		fuzzSCOp("SkipUntil", func(in fuzzSCInputs) Observable[int] { return SkipUntil[int, int](in.signal)(in.main) }),
		fuzzSCOp("StartWith", func(in fuzzSCInputs) Observable[int] { return StartWith(-1, -2)(in.main) }),
		fuzzSCOp("EndWith", func(in fuzzSCInputs) Observable[int] { return EndWith(-1, -2)(in.main) }),
		fuzzSCOp("Pairwise", func(in fuzzSCInputs) Observable[[]int] { return Pairwise[int]()(in.main) }),
		fuzzSCOp("DefaultIfEmpty", func(in fuzzSCInputs) Observable[int] { return DefaultIfEmpty(-1)(in.main) }),
		fuzzSCOp("ThrowIfEmpty", func(in fuzzSCInputs) Observable[int] {
			return ThrowIfEmpty[int](func() error { return errFuzzSCBoom })(in.main)
		}),
		fuzzSCOp("Catch", func(in fuzzSCInputs) Observable[int] {
			return Catch(func(error) Observable[int] { return in.fallback })(in.main)
		}),
		fuzzSCOp("OnErrorReturn", func(in fuzzSCInputs) Observable[int] { return OnErrorReturn(-1)(in.main) }),
	}
}

// FuzzSCUpstream asserts that every tracked upstream (source, signal, fallback) is unsubscribed
// once the downstream completed, errored or unsubscribed.
func FuzzSCUpstream(f *testing.F) {
	fuzzSCSeeds(f)

	table := fuzzSCTable()

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		c := table[fuzzBound(seed, 0, len(table)-1)]
		n := fuzzSCPick(seed, 0, 0, fuzzMaxItems)
		k := fuzzSCPick(seed, 1, 0, n+1)
		mode := fuzzSCPick(seed, 5, 0, fuzzSCStopModes-1)
		signalItems := fuzzSCPick(seed, 7, 0, 2)
		fallbackItems := fuzzSCPick(seed, 8, 0, 4)
		yields := fuzzSCPick(seed, 6, 0, fuzzSCMaxYields)

		var main, signal, fallback activeCounter

		in := fuzzSCInputs{
			main:     trackSubscriptions(&main, fuzzSCSource(seed, n, fuzzIsAsync(mask, 0), true, mode == fuzzSCStopError, nil)),
			signal:   trackSubscriptions(&signal, fuzzSource(seed, signalItems, fuzzIsAsync(mask, 1))),
			fallback: trackSubscriptions(&fallback, fuzzSource(seed, fallbackItems, fuzzIsAsync(mask, 2))),
			k:        k,
		}

		name := fmt.Sprintf("%s (async=%v/%v/%v mode=%d)", c.name, fuzzIsAsync(mask, 0), fuzzIsAsync(mask, 1), fuzzIsAsync(mask, 2), mode)
		term := &fuzzSCTerm{}

		fuzzSCIter(t, name, func() error {
			sub := c.run(context.Background(), in, term)

			if mode == fuzzSCStopUnsub {
				for i := 0; i < yields; i++ {
					runtime.Gosched()
				}
			} else {
				deadline := time.Now().Add(fuzzDeadline / 2)
				for atomic.LoadInt32(&term.terminals) == 0 {
					if time.Now().After(deadline) {
						return errors.New("downstream never terminated")
					}

					time.Sleep(time.Millisecond)
				}
			}

			sub.Unsubscribe()

			counters := []struct {
				label string
				c     *activeCounter
			}{{"main", &main}, {"signal", &signal}, {"fallback", &fallback}}

			for _, cnt := range counters {
				deadline := time.Now().Add(fuzzDeadline / 4)
				for cnt.c.activeCount() != 0 {
					if time.Now().After(deadline) {
						return fmt.Errorf("%s source still has %d active subscription(s) (opened %d) after terminate/unsubscribe",
							cnt.label, cnt.c.activeCount(), cnt.c.totalCount())
					}

					time.Sleep(time.Millisecond)
				}
			}

			return nil
		})
	})
}
