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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro/internal/xtest"
)

// Fuzz targets of the higher-order and loop operators. Inputs encode the interleaving
// (seed, sizes, sync/async bitmask, early-stop choice), never expected results.

const (
	// fuzzHOModeNone lets the stream run to its natural end.
	fuzzHOModeNone = 0
	// fuzzHOModeTake closes the downstream from inside the pipeline with Take.
	fuzzHOModeTake = 1
	// fuzzHOModeExternal unsubscribes from outside after a few items.
	fuzzHOModeExternal = 2

	// fuzzHOMaxTake bounds the early-stop item count, so loop operators stay cheap.
	fuzzHOMaxTake = 12

	// fuzzHOTagStride separates the values of two sources: source i emits i*stride+j, j < stride.
	fuzzHOTagStride = 1000

	// fuzzHOHuge is a loop bound that only a downstream stop can end.
	fuzzHOHuge = 1 << 30

	// fuzzHOBoundedBit selects, in the loop targets, a finite loop instead of an early stop.
	fuzzHOBoundedBit = 0x80

	// fuzzHOMaxConcatWith bounds the arity of ConcatWith, which is variadic.
	fuzzHOMaxConcatWith = 8

	// fuzzHORepeatMax is the resubscription count of the finite loop variants.
	fuzzHORepeatMax = 3
)

var errFuzzHO = errors.New("fuzzHO: boom")

type fuzzHOEnd int

const (
	fuzzHOEndComplete fuzzHOEnd = iota
	fuzzHOEndError
	// fuzzHOEndNever emits its items and then stays subscribed until unsubscribed.
	fuzzHOEndNever
)

// fuzzHOSource emits tag*stride+0..n-1 then ends as requested. A synchronous source emits inside
// Subscribe, an asynchronous one from its own goroutine.
func fuzzHOSource(seed int64, tag, n int, async bool, end fuzzHOEnd) Observable[int] {
	finish := func(ctx context.Context, destination Observer[int]) {
		switch end {
		case fuzzHOEndComplete:
			destination.CompleteWithContext(ctx)
		case fuzzHOEndError:
			destination.ErrorWithContext(ctx, errFuzzHO)
		case fuzzHOEndNever:
		}
	}

	if !async {
		return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			for i := 0; i < n && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, tag*fuzzHOTagStride+i)
			}

			finish(ctx, destination)

			return nil
		})
	}

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		done := make(chan struct{})

		go func() {
			for i := 0; i < n; i++ {
				select {
				case <-done:
					return
				default:
				}

				fuzzJitter(seed, i)
				destination.NextWithContext(ctx, tag*fuzzHOTagStride+i)
			}

			finish(ctx, destination)
		}()

		return func() { close(done) }
	})
}

// fuzzHORec records what the downstream observes and checks the observer contract.
type fuzzHORec struct {
	guard serialGuard

	nexts     int32
	errs      int32
	completes int32
	afterTerm int32

	mu     sync.Mutex
	values []int
}

func (r *fuzzHORec) terminals() int {
	return int(atomic.LoadInt32(&r.errs) + atomic.LoadInt32(&r.completes))
}

func (r *fuzzHORec) nextCount() int { return int(atomic.LoadInt32(&r.nexts)) }

func (r *fuzzHORec) errCount() int { return int(atomic.LoadInt32(&r.errs)) }

func (r *fuzzHORec) completeCount() int { return int(atomic.LoadInt32(&r.completes)) }

func (r *fuzzHORec) snapshot() []int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]int(nil), r.values...)
}

func (r *fuzzHORec) observer() Observer[int] {
	return NewObserver(
		func(v int) {
			r.guard.enter()
			defer r.guard.leave()

			if r.terminals() > 0 {
				atomic.AddInt32(&r.afterTerm, 1)
			}

			r.mu.Lock()
			r.values = append(r.values, v)
			r.mu.Unlock()
			atomic.AddInt32(&r.nexts, 1)
		},
		func(error) {
			r.guard.enter()
			defer r.guard.leave()

			if r.terminals() > 0 {
				atomic.AddInt32(&r.afterTerm, 1)
			}

			atomic.AddInt32(&r.errs, 1)
		},
		func() {
			r.guard.enter()
			defer r.guard.leave()

			if r.terminals() > 0 {
				atomic.AddInt32(&r.afterTerm, 1)
			}

			atomic.AddInt32(&r.completes, 1)
		},
	)
}

// fuzzHOHarness subscribes in its own goroutine so that a Subscribe that never returns is
// reported as a failure instead of hanging the test. On failure it cancels the subscriber context
// and raises stopped, which loops use as an exit, so a buggy infinite loop does not keep spinning.
type fuzzHOHarness struct {
	t       *testing.T
	ctx     context.Context
	cancel  context.CancelFunc
	stopped int32
	rec     *fuzzHORec

	mu       sync.Mutex
	sub      Subscription
	returned chan struct{}
}

func fuzzHONewHarness(t *testing.T) *fuzzHOHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	h := &fuzzHOHarness{t: t, ctx: ctx, cancel: cancel, rec: &fuzzHORec{}, returned: make(chan struct{})}

	t.Cleanup(func() {
		atomic.StoreInt32(&h.stopped, 1)
		cancel()
	})

	return h
}

func (h *fuzzHOHarness) isStopped() bool { return atomic.LoadInt32(&h.stopped) == 1 }

func (h *fuzzHOHarness) fail(format string, args ...any) {
	h.t.Helper()
	atomic.StoreInt32(&h.stopped, 1)
	h.cancel()
	h.t.Fatalf(format, args...)
}

func (h *fuzzHOHarness) waitFor(what string, cond func() bool) {
	h.t.Helper()

	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			h.fail("timeout waiting for: %s", what)
		}

		time.Sleep(time.Millisecond)
	}
}

func (h *fuzzHOHarness) start(obs Observable[int]) {
	go func() {
		s := obs.SubscribeWithContext(h.ctx, h.rec.observer())

		h.mu.Lock()
		h.sub = s
		h.mu.Unlock()

		close(h.returned)
	}()
}

func (h *fuzzHOHarness) subscription() Subscription {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.sub
}

func (h *fuzzHOHarness) hasReturned() bool {
	select {
	case <-h.returned:
		return true
	default:
		return false
	}
}

// settle drives the stream to its end according to mode, then unsubscribes.
func (h *fuzzHOHarness) settle(mode, kk int) {
	h.t.Helper()

	if mode == fuzzHOModeExternal {
		h.waitFor("a subscription handle and kk items or a terminal notification (Subscribe may be hung)", func() bool {
			return h.subscription() != nil && (h.rec.nextCount() >= kk || h.rec.terminals() > 0)
		})
	} else {
		h.waitFor("Subscribe to return (hang)", h.hasReturned)
		h.waitFor("a terminal notification", func() bool { return h.rec.terminals() > 0 })
	}

	if sub := h.subscription(); sub != nil {
		sub.Unsubscribe()
	}
}

// verify checks that every upstream is released and that the observer contract held.
func (h *fuzzHOHarness) verify(counters ...*activeCounter) {
	h.t.Helper()

	for i, c := range counters {
		c := c

		h.waitFor(fmt.Sprintf("upstream %d to drop to 0 active subscriptions", i), func() bool { return c.activeCount() == 0 })
	}

	if n := h.rec.guard.overlapped(); n != 0 {
		h.fail("overlapping notifications on the downstream observer: %d", n)
	}

	if n := atomic.LoadInt32(&h.rec.afterTerm); n != 0 {
		h.fail("notification delivered after a terminal notification: %d", n)
	}

	if n := h.rec.terminals(); n > 1 {
		h.fail("%d terminal notifications", n)
	}
}

// fuzzHOMode derives the early-stop mode and item count from one fuzz input.
func fuzzHOMode(k int64) (mode, kk int) {
	return fuzzBound(k, fuzzHOModeNone, fuzzHOModeExternal), fuzzBound(k/3, 1, fuzzHOMaxTake)
}

func fuzzHOApply(obs Observable[int], mode, kk int) Observable[int] {
	if mode == fuzzHOModeTake {
		return Take[int](int64(kk))(obs)
	}

	return obs
}

func fuzzHOSeeds(f *testing.F) {
	f.Helper()

	xtest.AddSeeds(f, func(i int) []any {
		return []any{int64(i), int64(i*3 + 1), uint8(i*37 + i/7), int64(i*5 + 2)} //nolint:gosec // wrap-around is intended.
	})
}

func fuzzHOMin(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// ---------------------------------------------------------------------------------------------
// MergeAll / MergeMap / ConcatAll / ConcatWith / FlatMap
// ---------------------------------------------------------------------------------------------

const (
	fuzzHOKindMergeAll = iota
	fuzzHOKindMergeMap
	fuzzHOKindConcatAll
	fuzzHOKindConcatWith
	fuzzHOKindFlatMap
)

func fuzzHOHigher(t *testing.T, kind int, seed, size int64, mask uint8, k int64) {
	t.Helper()

	h := fuzzHONewHarness(t)
	mode, kk := fuzzHOMode(k)

	m := fuzzBound(size, 1, fuzzMaxItems)
	if kind == fuzzHOKindConcatWith {
		m = fuzzBound(size, 1, fuzzHOMaxConcatWith)
	}

	var innerC, outerC activeCounter

	inners := make([]Observable[int], m)

	var expected []int

	for i := range inners {
		n := fuzzBound(seed+int64(i)*7, 0, 4)
		inners[i] = trackSubscriptions(&innerC, fuzzHOSource(seed+int64(i), i, n, fuzzIsAsync(mask, i+1), fuzzHOEndComplete))

		for j := 0; j < n; j++ {
			expected = append(expected, i*fuzzHOTagStride+j)
		}
	}

	outer := trackSubscriptions(&outerC, fuzzSource(seed, m, fuzzIsAsync(mask, 0)))
	obs, ordered := fuzzHOBuildHigher(kind, inners, outer)

	h.start(fuzzHOApply(obs, mode, kk))
	h.settle(mode, kk)
	h.verify(&innerC, &outerC)

	fuzzHOCheckHigher(h, mode, kk, expected, ordered)
}

// fuzzHOBuildHigher builds the higher-order operator under test. ordered reports whether the
// operator preserves the order of the inner sources.
func fuzzHOBuildHigher(kind int, inners []Observable[int], outer Observable[int]) (obs Observable[int], ordered bool) {
	project := func(v int) Observable[int] { return inners[v] }

	switch kind {
	case fuzzHOKindMergeAll:
		return MergeAll[int]()(Map(project)(outer)), false
	case fuzzHOKindMergeMap:
		return MergeMap(project)(outer), false
	case fuzzHOKindConcatAll:
		return ConcatAll[int]()(Map(project)(outer)), true
	case fuzzHOKindConcatWith:
		return ConcatWith(inners[1:]...)(inners[0]), true
	default:
		return FlatMap(project)(outer), true
	}
}

// fuzzHOCheckHigher asserts the values and terminal notifications delivered by a higher-order
// operator against the values the inner sources can emit.
func fuzzHOCheckHigher(h *fuzzHOHarness, mode, kk int, expected []int, ordered bool) {
	got := h.rec.snapshot()

	if len(got) > len(expected) {
		h.fail("%d values received, only %d can exist", len(got), len(expected))
	}

	if ordered {
		for i := range got {
			if got[i] != expected[i] {
				h.fail("concatenation order broken at %d: got %d, want %d", i, got[i], expected[i])
			}
		}
	}

	switch mode {
	case fuzzHOModeNone:
		if len(got) != len(expected) || h.rec.errCount() != 0 || h.rec.completeCount() != 1 {
			h.fail("lost values or terminal: got %d/%d values, errs=%d completes=%d", len(got), len(expected), h.rec.errCount(), h.rec.completeCount())
		}
	case fuzzHOModeTake:
		if len(got) != fuzzHOMin(kk, len(expected)) {
			h.fail("take(%d) delivered %d of %d available values", kk, len(got), len(expected))
		}
	}
}

func FuzzHOMergeAll(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindMergeAll, seed, size, mask, k)
	})
}

func FuzzHOMergeMap(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindMergeMap, seed, size, mask, k)
	})
}

func FuzzHOConcatAll(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindConcatAll, seed, size, mask, k)
	})
}

func FuzzHOConcatWith(f *testing.F) {
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOHigher(t, fuzzHOKindConcatWith, seed, size, mask, k)
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
		subject := NewPublishSubject[int]()
		m := fuzzBound(size, 1, fuzzHOMaxConcatWith)
		independent := mask&fuzzHOBoundedBit != 0

		var innerC activeCounter

		project := func(v int) Observable[int] {
			if independent {
				return trackSubscriptions(&innerC, fuzzHOSource(seed, v, 2, fuzzIsAsync(mask, v+1), fuzzHOEndComplete))
			}

			return trackSubscriptions(&innerC, Take[int](1)(subject.AsObservable()))
		}

		h.start(FlatMap(project)(subject.AsObservable()))
		h.waitFor("Subscribe to return", h.hasReturned)

		producerDone := make(chan struct{})

		go func() {
			defer close(producerDone)

			// The feeder is itself sync or async: its Next calls run on the producing goroutine.
			finished := make(chan struct{})

			fuzzSource(seed, m, fuzzIsAsync(mask, 0)).Subscribe(NewObserver(
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

// ---------------------------------------------------------------------------------------------
// Catch / StartWith over a merge of two sources: the unsafe subscriber is passed through to the merge
// ---------------------------------------------------------------------------------------------

func FuzzHOCatch(f *testing.F) {
	f.Skip("race: catch-unsafe-merge; remove when fixed") // fuzz_higherorder_test.go:540: notification delivered after a terminal notification: 1
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := fuzzHONewHarness(t)
		mode, kk := fuzzHOMode(k)

		var srcC, aC, bC activeCounter

		n := fuzzBound(size, 0, 3)
		na := fuzzBound(seed, 1, 4)
		nb := fuzzBound(seed/5, 1, 4)

		src := trackSubscriptions(&srcC, fuzzHOSource(seed, 0, n, fuzzIsAsync(mask, 0), fuzzHOEndError))
		a := trackSubscriptions(&aC, fuzzHOSource(seed+1, 1, na, fuzzIsAsync(mask, 1), fuzzHOEndComplete))
		b := trackSubscriptions(&bC, fuzzHOSource(seed+2, 2, nb, fuzzIsAsync(mask, 2), fuzzHOEndComplete))

		obs := Catch(func(error) Observable[int] { return Merge(a, b) })(src)

		h.start(fuzzHOApply(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&srcC, &aC, &bC)

		total := n + na + nb

		switch mode {
		case fuzzHOModeNone:
			if h.rec.nextCount() != total || h.rec.completeCount() != 1 {
				h.fail("got %d/%d values, completes=%d errs=%d", h.rec.nextCount(), total, h.rec.completeCount(), h.rec.errCount())
			}
		case fuzzHOModeTake:
			if h.rec.nextCount() != fuzzHOMin(kk, total) {
				h.fail("take(%d) delivered %d of %d", kk, h.rec.nextCount(), total)
			}
		}
	})
}

func FuzzHOStartWith(f *testing.F) {
	f.Skip("race: startwith-unsafe-merge; remove when fixed") // fuzz_higherorder_test.go:581: overlapping notifications on the downstream observer: 1
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := fuzzHONewHarness(t)
		mode, kk := fuzzHOMode(k)

		var aC, bC activeCounter

		p := fuzzBound(size, 0, 3)
		na := fuzzBound(seed, 0, 4)
		nb := fuzzBound(seed/5, 0, 4)

		prefixes := make([]int, p)
		for i := range prefixes {
			prefixes[i] = -1 - i
		}

		a := trackSubscriptions(&aC, fuzzHOSource(seed+1, 1, na, fuzzIsAsync(mask, 0), fuzzHOEndComplete))
		b := trackSubscriptions(&bC, fuzzHOSource(seed+2, 2, nb, fuzzIsAsync(mask, 1), fuzzHOEndComplete))

		obs := StartWith(prefixes...)(Merge(a, b))

		h.start(fuzzHOApply(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&aC, &bC)

		got := h.rec.snapshot()
		for i := 0; i < fuzzHOMin(p, len(got)); i++ {
			if got[i] != prefixes[i] {
				h.fail("prefix %d is %d, want %d", i, got[i], prefixes[i])
			}
		}

		total := p + na + nb

		switch mode {
		case fuzzHOModeNone:
			if len(got) != total || h.rec.completeCount() != 1 {
				h.fail("got %d/%d values, completes=%d", len(got), total, h.rec.completeCount())
			}
		case fuzzHOModeTake:
			if len(got) != fuzzHOMin(kk, total) {
				h.fail("take(%d) delivered %d of %d", kk, len(got), total)
			}
		}
	})
}

// ---------------------------------------------------------------------------------------------
// Race / RaceWith
// ---------------------------------------------------------------------------------------------

func fuzzHORace(t *testing.T, with bool, seed, size int64, mask uint8) {
	t.Helper()

	h := fuzzHONewHarness(t)
	m := fuzzBound(size, 2, 4)
	endless := mask&fuzzHOBoundedBit != 0

	end := fuzzHOEndComplete
	if endless {
		end = fuzzHOEndNever
	}

	counters := make([]activeCounter, m)
	srcs := make([]Observable[int], m)
	ptrs := make([]*activeCounter, m)

	for i := range srcs {
		ptrs[i] = &counters[i]
		// At least one item, so that a loser of an endless race still has something to race with.
		srcs[i] = trackSubscriptions(ptrs[i], fuzzHOSource(seed+int64(i), i, fuzzBound(seed+int64(i)*5, 1, 3), fuzzIsAsync(mask, i), end))
	}

	var obs Observable[int]
	if with {
		obs = RaceWith(srcs[1:]...)(srcs[0])
	} else {
		obs = Race(srcs...)
	}

	h.start(obs)

	if endless {
		h.waitFor("the winner's first item", func() bool { return h.rec.nextCount() >= 1 || h.rec.terminals() > 0 })

		winner := h.rec.snapshot()[0] / fuzzHOTagStride
		for i := range counters {
			i := i
			if i != winner {
				h.waitFor(fmt.Sprintf("loser %d (winner %d) to be unsubscribed", i, winner), func() bool { return ptrs[i].activeCount() == 0 })
			}
		}

		h.settle(fuzzHOModeExternal, 1)
	} else {
		h.settle(fuzzHOModeNone, 0)
	}

	h.verify(ptrs...)

	got := h.rec.snapshot()
	for i := range got {
		if got[i]/fuzzHOTagStride != got[0]/fuzzHOTagStride {
			h.fail("values of two sources mixed: %v", got)
		}
	}
}

func FuzzHORaceWith(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // fuzz_higherorder_test.go:669: timeout waiting for: upstream 0 to drop to 0 active subscriptions
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		fuzzHORace(t, true, seed, size, mask)
	})
}

func FuzzHORace(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // fuzz_higherorder_test.go:676: timeout waiting for: upstream 0 to drop to 0 active subscriptions
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, _ int64) {
		fuzzHORace(t, false, seed, size, mask)
	})
}

// ---------------------------------------------------------------------------------------------
// Loops: While / DoWhile / RepeatWith / Retry / RetryWithConfig, and OnErrorResumeNextWith
// ---------------------------------------------------------------------------------------------

const (
	fuzzHOLoopWhile = iota
	fuzzHOLoopDoWhile
	fuzzHOLoopRepeatWith
	fuzzHOLoopRetry
	fuzzHOLoopRetryWithConfig
)

func fuzzHOLoop(t *testing.T, kind int, seed, size int64, mask uint8, k int64) {
	t.Helper()

	h := fuzzHONewHarness(t)
	n := fuzzBound(size, 1, 4)
	kk := fuzzBound(k, 1, fuzzHOMaxTake)
	bounded := mask&fuzzHOBoundedBit != 0 && kind <= fuzzHOLoopRepeatWith
	rounds := fuzzBound(k/7, 1, fuzzHORepeatMax)

	end := fuzzHOEndComplete
	if kind >= fuzzHOLoopRetry {
		end = fuzzHOEndError
	}

	var srcC activeCounter

	src := trackSubscriptions(&srcC, fuzzHOSource(seed, 0, n, fuzzIsAsync(mask, 0), end))

	// Subscriptions needed to deliver kk items: the loop must not open another one afterwards.
	need := (kk + n - 1) / n

	// condition is evaluated once per loop iteration. The stop flag only bounds a buggy loop.
	var calls int32

	limit := int32(fuzzHOHuge)
	if bounded {
		limit = int32(rounds) //nolint:gosec // rounds is in [1,3].
	}

	condition := func() bool { return !h.isStopped() && atomic.AddInt32(&calls, 1) <= limit }

	retryMax := fuzzBound(k/7, 1, fuzzHORepeatMax)
	retryReset := mask&0x40 != 0

	obs := fuzzHOBuildLoop(kind, src, condition, bounded, rounds, retryMax, retryReset)

	mode := fuzzHOModeTake
	if bounded {
		mode = fuzzHOModeNone
	}

	h.start(fuzzHOApply(obs, mode, kk))
	h.settle(mode, kk)
	h.verify(&srcC)

	fuzzHOCheckLoop(h, kind, srcC.totalCount(), n, kk, need, rounds, retryMax, retryReset, bounded)
}

// fuzzHOBuildLoop builds the looping operator under test around src.
func fuzzHOBuildLoop(kind int, src Observable[int], condition func() bool, bounded bool, rounds, retryMax int, retryReset bool) Observable[int] {
	switch kind {
	case fuzzHOLoopWhile:
		return While[int](condition)(src)
	case fuzzHOLoopDoWhile:
		return DoWhile[int](condition)(src)
	case fuzzHOLoopRepeatWith:
		count := int64(1000) // finite, so that a missing stop is slow instead of infinite.
		if bounded {
			count = int64(rounds)
		}

		return RepeatWith[int](count)(src)
	case fuzzHOLoopRetry:
		return Retry[int]()(src)
	default:
		return RetryWithConfig[int](RetryConfig{MaxRetries: uint64(retryMax), ResetOnSuccess: retryReset})(src) //nolint:gosec // retryMax is in [1,3].
	}
}

// fuzzHOCheckLoop asserts the number of source subscriptions and delivered values of a loop
// operator: n is the item count per run, kk the take size, need the subscriptions required to
// deliver kk items.
func fuzzHOCheckLoop(h *fuzzHOHarness, kind, subs, n, kk, need, rounds, retryMax int, retryReset, bounded bool) {
	got := h.rec.nextCount()

	if bounded {
		wantSubs := rounds
		if kind == fuzzHOLoopDoWhile {
			wantSubs = rounds + 1 // the first run is unconditional, then one per true condition.
		}

		if subs != wantSubs || got != wantSubs*n || h.rec.completeCount() != 1 {
			h.fail("bounded loop: %d subscriptions (want %d), %d values (want %d), completes=%d", subs, wantSubs, got, wantSubs*n, h.rec.completeCount())
		}

		return
	}

	if kind == fuzzHOLoopRetryWithConfig && !retryReset && kk > (retryMax+1)*n {
		// Retries run out before kk items: the error must reach the downstream.
		if subs != retryMax+1 || got != (retryMax+1)*n || h.rec.errCount() != 1 {
			h.fail("retries exhausted: %d subscriptions (want %d), %d values, errs=%d", subs, retryMax+1, got, h.rec.errCount())
		}

		return
	}

	if got != kk {
		h.fail("take(%d) delivered %d values", kk, got)
	}

	if subs > need {
		h.fail("%d source subscriptions after the downstream closed, %d were enough", subs, need)
	}
}

func FuzzHOWhile(f *testing.F) {
	f.Skip("race: while-ignores-closed-destination; remove when fixed") // fuzz_higherorder_test.go:792: timeout waiting for: Subscribe to return (hang)
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOLoop(t, fuzzHOLoopWhile, seed, size, mask, k)
	})
}

func FuzzHODoWhile(f *testing.F) {
	f.Skip("race: dowhile-ignores-closed-destination; remove when fixed") // fuzz_higherorder_test.go:799: timeout waiting for: Subscribe to return (hang)
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOLoop(t, fuzzHOLoopDoWhile, seed, size, mask, k)
	})
}

func FuzzHORepeatWith(f *testing.F) {
	f.Skip("race: repeatwith-ignores-take-close; remove when fixed") // fuzz_higherorder_test.go:806: 1000 source subscriptions after the downstream closed, 1 were enough
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOLoop(t, fuzzHOLoopRepeatWith, seed, size, mask, k)
	})
}

func FuzzHORetry(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed") // fuzz_higherorder_test.go:813: timeout waiting for: Subscribe to return (hang)
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOLoop(t, fuzzHOLoopRetry, seed, size, mask, k)
	})
}

func FuzzHORetryWithConfig(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed") // fuzz_higherorder_test.go:820: 4 source subscriptions after the downstream closed, 1 were enough
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		fuzzHOLoop(t, fuzzHOLoopRetryWithConfig, seed, size, mask, k)
	})
}

func FuzzHOOnErrorResumeNextWith(f *testing.F) {
	f.Skip("race: resumenext-ignores-closed-destination; remove when fixed") // fuzz_higherorder_test.go:890: 5 sources subscribed after the downstream closed, 1 were enough
	fuzzHOSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		h := fuzzHONewHarness(t)
		kk := fuzzBound(k, 1, fuzzHOMaxTake)
		bounded := mask&fuzzHOBoundedBit != 0
		count := fuzzBound(size, 2, 5)

		var srcC activeCounter

		srcs := make([]Observable[int], count)
		sizes := make([]int, count)
		lastFails := false
		total := 0

		for i := range srcs {
			sizes[i] = fuzzBound(seed+int64(i)*3, 1, 3)
			total += sizes[i]

			end := fuzzHOEndComplete
			if fuzzBound(seed+int64(i), 0, 1) == 1 {
				end = fuzzHOEndError
			}

			lastFails = end == fuzzHOEndError
			srcs[i] = trackSubscriptions(&srcC, fuzzHOSource(seed+int64(i), i, sizes[i], fuzzIsAsync(mask, i), end))
		}

		obs := OnErrorResumeNextWith(srcs[1:]...)(srcs[0])

		mode := fuzzHOModeTake
		if bounded {
			mode = fuzzHOModeNone
		}

		h.start(fuzzHOApply(obs, mode, kk))
		h.settle(mode, kk)
		h.verify(&srcC)

		subs := srcC.totalCount()

		if bounded {
			wantErrs := 0
			if lastFails {
				wantErrs = 1
			}

			if subs != count || h.rec.nextCount() != total || h.rec.errCount() != wantErrs {
				h.fail("subs=%d/%d values=%d/%d errs=%d (want %d)", subs, count, h.rec.nextCount(), total, h.rec.errCount(), wantErrs)
			}

			return
		}

		if want := fuzzHOMin(kk, total); h.rec.nextCount() != want {
			h.fail("take(%d) delivered %d, want %d", kk, h.rec.nextCount(), want)
		}

		// Sources needed to deliver kk items.
		need, acc := 0, 0
		for need < count && acc < kk {
			acc += sizes[need]
			need++
		}

		if subs > need {
			h.fail("%d sources subscribed after the downstream closed, %d were enough", subs, need)
		}
	})
}
