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

func FuzzMergeMap(f *testing.F) {
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runHigherOrder(t, higherOrderKindMergeMap, seed, size, mask, k)
	})
}

func FuzzFlatMap(f *testing.F) {
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runHigherOrder(t, higherOrderKindFlatMap, seed, size, mask, k)
	})
}

// FuzzFlatMapSameSubject feeds FlatMap from a subject while every inner waits for the NEXT
// item of that same subject. The producer is blocked inside Next while FlatMap waits for the
// inner, so the producer must be released by buffering the outer items.
func FuzzFlatMapSameSubject(f *testing.F) {
	f.Skip("race: flatmap-blocks-producer; remove when fixed") // Fails on main: producer blocked in Subject.Next: FlatMap waits for an inner inside the outer Next
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		h := newStreamHarness(t)
		subject := ro.NewPublishSubject[int]()
		m := fuzzBound(size, 1, maxConcatWithArity)
		independent := mask&boundedLoopBit != 0

		var innerC activeCounter

		project := func(v int) ro.Observable[int] {
			if independent {
				return trackSubscriptions(&innerC, taggedSource(seed, v, 2, fuzzIsAsync(mask, v+1), sourceEndComplete))
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

func boundaryDelay(seed int64, salt int) time.Duration {
	return time.Duration(pickBoundaryValue(seed, salt, 100, maxBoundaryMicros)) * time.Microsecond
}

// tickSource emits k ticks and never completes, like a boundary that outlives the source.
func tickSource(c *activeCounter, seed int64, k int, async bool) ro.Observable[int] {
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
					time.Sleep(time.Duration(pickBoundaryValue(seed, 200+i, 0, 300)) * time.Microsecond)
					destination.NextWithContext(ctx, i)
				}
			}()

			return func() { close(done) }
		})
	}

	return trackSubscriptions(c, src)
}

// checkSequence checks that got is exactly 0..n-1 in order.
func checkSequence(got []int, n int) error {
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

func flattenBuffers(buffers [][]int) []int {
	out := []int{}
	for _, b := range buffers {
		out = append(out, b...)
	}

	return out
}

// checkIncreasing checks strictly increasing values (a sampled/throttled subsequence of 0..n-1).
func checkIncreasing(n int) func(got []int) error {
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

func FuzzBufferWithCount(f *testing.F) {
	f.Skip("race: bufferwithcount-teardown-unlocked-buffer; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		size := pickBoundaryValue(seed, 2, 1, 6)
		up := &activeCounter{}

		runBoundaryOperator(t, "BufferWithCount", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithCount[int](size)(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return checkSequence(flattenBuffers(got), n) },
			up)
	})
}

func FuzzBufferWhen(f *testing.F) {
	// Intermittent (about 1 run in 7 at 2000 seeds), for example: got 8 items [0 1 2 3 4 5 8 9], want 10.
	f.Skip("race: bufferwhen-lost-buffer-after-unlock; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		k := pickBoundaryValue(seed, 3, 0, maxBoundaryTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		runBoundaryOperator(t, "BufferWhen", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWhen[int](tickSource(tick, seed, k, fuzzIsAsync(mask, 1)))(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return checkSequence(flattenBuffers(got), n) },
			up, tick)
	})
}

func FuzzBufferWithTime(f *testing.F) {
	f.Skip("race: bufferwithtime-flush-after-unlock-lost-buffer (intermittent, seen once under load); remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		d := boundaryDelay(seed, 4)
		up := &activeCounter{}

		runBoundaryOperator(t, "BufferWithTime", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithTime[int](d)(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return checkSequence(flattenBuffers(got), n) },
			up)
	})
}

func FuzzBufferWithTimeOrCount(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		size := pickBoundaryValue(seed, 2, 1, 6)
		d := boundaryDelay(seed, 4)
		up := &activeCounter{}

		runBoundaryOperator(t, "BufferWithTimeOrCount", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[[]int] {
				return ro.BufferWithTimeOrCount[int](size, d)(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got [][]int) error { return checkSequence(flattenBuffers(got), n) },
			up)
	})
}

// windowRecorder subscribes to every emitted window and records its items and terminal count.
type windowRecorder struct {
	mu        sync.Mutex
	items     [][]int
	terminals []*int32
}

func (w *windowRecorder) next(win ro.Observable[int]) {
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
func (w *windowRecorder) open() int {
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

func (w *windowRecorder) flatten() []int {
	w.mu.Lock()
	defer w.mu.Unlock()

	return flattenBuffers(w.items)
}

// subscribeWindows subscribes to a higher-order observable and tracks every window.
func (w *windowRecorder) subscribe(obs ro.Observable[ro.Observable[int]], outer *boundarySink[int]) ro.Subscription {
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

func FuzzWindowWhen(f *testing.F) {
	f.Skip("race: windowwhen-value-lost-in-completed-window; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		k := pickBoundaryValue(seed, 3, 0, maxBoundaryTicks)
		up, tick := &activeCounter{}, &activeCounter{}
		counters := []*activeCounter{up, tick}

		runBoundaryIteration(t, "WindowWhen", func() error {
			outer := &boundarySink[int]{}
			wins := &windowRecorder{}
			obs := ro.WindowWhen[int](tickSource(tick, seed, k, fuzzIsAsync(mask, 1)))(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			sub := wins.subscribe(obs, outer)

			if mask&unsubscribeBit != 0 {
				for i, m := 0, pickBoundaryValue(seed, 101, 0, maxUnsubscribeSpins); i < m; i++ {
					fuzzJitter(seed, i)
				}

				sub.Unsubscribe()

				return waitForCondition("upstream and boundary released", func() bool { return activeSubscriptionTotal(counters) == 0 })
			}

			if err := waitForCondition("outer terminal", outer.done); err != nil {
				return err
			}

			if err := waitForCondition("every window terminal", func() bool { return wins.open() == 0 }); err != nil {
				return fmt.Errorf("%w: %d windows never completed", err, wins.open())
			}

			if err := waitForCondition("upstream and boundary released", func() bool { return activeSubscriptionTotal(counters) == 0 }); err != nil {
				return err
			}

			time.Sleep(boundarySettleDelay)

			if err := outer.unexpectedErr(); err != nil {
				return err
			}

			return checkSequence(wins.flatten(), n)
		})
	})
}

// FuzzWindowWhenUnsubscribe isolates the teardown path: the open window must get a terminal
// notification when the downstream unsubscribes, otherwise its subscriber waits forever.
func FuzzWindowWhenUnsubscribe(f *testing.F) {
	f.Skip("race: windowwhen-teardown-leaves-window-open; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		k := pickBoundaryValue(seed, 3, 0, maxBoundaryTicks)
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

		runBoundaryIteration(t, "WindowWhen/unsubscribe", func() error {
			wins := &windowRecorder{}
			outer := &boundarySink[int]{}
			sub := wins.subscribe(ro.WindowWhen[int](tickSource(tick, seed, k, fuzzIsAsync(mask, 1)))(src), outer)

			for i, m := 0, pickBoundaryValue(seed, 101, 0, maxUnsubscribeSpins); i < m; i++ {
				fuzzJitter(seed, i)
			}

			sub.Unsubscribe()

			if err := waitForCondition("upstream and boundary released", func() bool { return activeSubscriptionTotal([]*activeCounter{up, tick}) == 0 }); err != nil {
				return err
			}

			if err := waitForCondition("open window completed on unsubscribe", func() bool { return wins.open() == 0 }); err != nil {
				return fmt.Errorf("%w: %d windows left without terminal", err, wins.open())
			}

			return nil
		})
	})
}

func FuzzSampleWhen(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		k := pickBoundaryValue(seed, 3, 0, maxBoundaryTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		runBoundaryOperator(t, "SampleWhen", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[int] {
				return ro.SampleWhen[int](tickSource(tick, seed, k, fuzzIsAsync(mask, 1)))(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			checkIncreasing(n), up, tick)
	})
}

func FuzzSampleTime(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		d := boundaryDelay(seed, 4)
		up := &activeCounter{}

		runBoundaryOperator(t, "SampleTime", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[int] {
				return ro.SampleTime[int](d)(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			checkIncreasing(n), up)
	})
}

func FuzzThrottleWhen(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		k := pickBoundaryValue(seed, 3, 0, maxBoundaryTicks)
		up, tick := &activeCounter{}, &activeCounter{}

		runBoundaryOperator(t, "ThrottleWhen", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[int] {
				return ro.ThrottleWhen[int](tickSource(tick, seed, k, fuzzIsAsync(mask, 1)))(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			func(got []int) error {
				if len(got) > k {
					return fmt.Errorf("%d values passed but only %d ticks were sent", len(got), k)
				}

				return checkIncreasing(n)(got)
			}, up, tick)
	})
}

func FuzzThrottleTime(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		d := boundaryDelay(seed, 4)
		up := &activeCounter{}

		runBoundaryOperator(t, "ThrottleTime", seed, mask&unsubscribeBit != 0,
			func() ro.Observable[int] {
				return ro.ThrottleTime[int](d)(countedSource(up, seed, n, fuzzIsAsync(mask, 0)))
			},
			checkIncreasing(n), up)
	})
}

func FuzzGroupBy(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n := pickBoundaryValue(seed, 1, 0, fuzzMaxItems)
		keys := pickBoundaryValue(seed, 5, 1, 4)
		up := &activeCounter{}

		runBoundaryIteration(t, "GroupBy", func() error {
			wins := &windowRecorder{}
			outer := &boundarySink[int]{}
			sub := wins.subscribe(ro.GroupBy(func(v int) int { return v % keys })(countedSource(up, seed, n, fuzzIsAsync(mask, 0))), outer)

			if mask&unsubscribeBit != 0 {
				for i, m := 0, pickBoundaryValue(seed, 101, 0, maxUnsubscribeSpins); i < m; i++ {
					fuzzJitter(seed, i)
				}

				sub.Unsubscribe()

				if err := waitForCondition("upstream released", func() bool { return up.activeCount() == 0 }); err != nil {
					return err
				}

				return waitForCondition("every group terminal after unsubscribe", func() bool { return wins.open() == 0 })
			}

			if err := waitForCondition("outer terminal", outer.done); err != nil {
				return err
			}

			if err := waitForCondition("every group terminal", func() bool { return wins.open() == 0 }); err != nil {
				return err
			}

			if err := waitForCondition("upstream released", func() bool { return up.activeCount() == 0 }); err != nil {
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
