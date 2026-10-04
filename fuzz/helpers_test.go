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
	"errors"
	"fmt"
	"runtime"
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
	"go.uber.org/goleak"
)

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

// Helpers shared by the Fuzz* targets. They use sync/atomic
// functions instead of atomic.Int32 because go.mod declares go 1.18.

const (
	// fuzzDeadline bounds every blocking wait of a fuzz iteration. A hang is a bug, so the
	// bound is generous enough for a loaded CI machine but short enough to fail fast.
	fuzzDeadline = 5 * time.Second

	// fuzzMaxItems bounds the number of items a fuzz input can request, to keep one iteration fast.
	fuzzMaxItems = 64

	// fuzzMaxGoroutines bounds the number of concurrent actors a fuzz input can request.
	fuzzMaxGoroutines = 8
)

// fuzzBound maps any fuzz input to [lo, hi], so that negative or huge values stay valid scenarios.
func fuzzBound(v int64, lo, hi int) int {
	span := int64(hi - lo + 1)

	r := v % span
	if r < 0 {
		r = -r
	}

	return lo + int(r)
}

// fuzzJitter yields the processor at positions chosen by seed, so that different seeds
// explore different interleavings. step identifies the call site inside one iteration.
func fuzzJitter(seed int64, step int) {
	// Mix seed and step so neighbouring steps do not yield together.
	mixed := uint64(seed)*6364136223846793005 + uint64(step)*1442695040888963407 //nolint:gosec // wrap-around is intended.

	switch mixed >> 61 { // 3 top bits: 0..7.
	case 0, 1:
		runtime.Gosched()
	case 2:
		time.Sleep(time.Microsecond)
	}
}

// serialGuard detects overlapping calls: wrap the body of an observer callback with
// enter/leave and check overlapped() at the end of the test.
type serialGuard struct {
	inside     int32
	overlapCnt int32
}

func (g *serialGuard) enter() {
	if atomic.AddInt32(&g.inside, 1) > 1 {
		atomic.AddInt32(&g.overlapCnt, 1)
	}
}

func (g *serialGuard) leave() { atomic.AddInt32(&g.inside, -1) }

func (g *serialGuard) overlapped() int { return int(atomic.LoadInt32(&g.overlapCnt)) }

// activeCounter counts live subscriptions to a source. Use it with Defer + TapOnSubscribe-like
// bookkeeping to assert that an operator unsubscribed from every upstream.
type activeCounter struct {
	active int32
	total  int32
}

func (c *activeCounter) open() {
	atomic.AddInt32(&c.active, 1)
	atomic.AddInt32(&c.total, 1)
}

func (c *activeCounter) close() { atomic.AddInt32(&c.active, -1) }

func (c *activeCounter) activeCount() int { return int(atomic.LoadInt32(&c.active)) }

func (c *activeCounter) totalCount() int { return int(atomic.LoadInt32(&c.total)) }

// track wraps source so that every subscription increments the counter until it is torn down.
func trackSubscriptions[T any](c *activeCounter, source ro.Observable[T]) ro.Observable[T] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[T]) ro.Teardown {
		c.open()

		sub := source.SubscribeWithContext(ctx, destination)

		return func() {
			sub.Unsubscribe()
			c.close()
		}
	})
}

// fuzzSource emits 0..n-1 then completes. Every fuzz target must run against BOTH kinds:
// a synchronous source emits inside Subscribe (teardown is registered only after it returns),
// while an asynchronous source emits from its own goroutine (Next races Unsubscribe and Complete).
// Derive async from a bit of the fuzz input so the corpus covers both.
func fuzzSource(seed int64, n int, async bool) ro.Observable[int] {
	if !async {
		return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
			for i := 0; i < n && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, i)
			}

			destination.CompleteWithContext(ctx)

			return nil
		})
	}

	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
		done := make(chan struct{})

		go func() {
			for i := 0; i < n; i++ {
				select {
				case <-done:
					return
				default:
				}

				fuzzJitter(seed, i)
				destination.NextWithContext(ctx, i)
			}

			destination.CompleteWithContext(ctx)
		}()

		return func() { close(done) }
	})
}

// fuzzIsAsync reads the sync/async choice of source number idx from a bitmask fuzz input.
func fuzzIsAsync(mask uint8, idx int) bool { return mask&(1<<uint(idx%8)) != 0 }

// fuzzWaitFor polls cond until it is true or fuzzDeadline expires.
func fuzzWaitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()

	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timeout waiting for: %s", what)
		}

		time.Sleep(time.Millisecond)
	}
}

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
func fuzzHOSource(seed int64, tag, n int, async bool, end fuzzHOEnd) ro.Observable[int] {
	finish := func(ctx context.Context, destination ro.Observer[int]) {
		switch end {
		case fuzzHOEndComplete:
			destination.CompleteWithContext(ctx)
		case fuzzHOEndError:
			destination.ErrorWithContext(ctx, errFuzzHO)
		case fuzzHOEndNever:
		}
	}

	if !async {
		return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
			for i := 0; i < n && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, tag*fuzzHOTagStride+i)
			}

			finish(ctx, destination)

			return nil
		})
	}

	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
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

func (r *fuzzHORec) observer() ro.Observer[int] {
	return ro.NewObserver(
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
	sub      ro.Subscription
	returned chan struct{}
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

func (h *fuzzHOHarness) start(obs ro.Observable[int]) {
	go func() {
		s := obs.SubscribeWithContext(h.ctx, h.rec.observer())

		h.mu.Lock()
		h.sub = s
		h.mu.Unlock()

		close(h.returned)
	}()
}

func (h *fuzzHOHarness) subscription() ro.Subscription {
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

// fuzzHOMode derives the early-stop mode and item count from one fuzz input.
func fuzzHOMode(k int64) (mode, kk int) {
	return fuzzBound(k, fuzzHOModeNone, fuzzHOModeExternal), fuzzBound(k/3, 1, fuzzHOMaxTake)
}

func fuzzHOApply(obs ro.Observable[int], mode, kk int) ro.Observable[int] {
	if mode == fuzzHOModeTake {
		return ro.Take[int](int64(kk))(obs)
	}

	return obs
}

func fuzzHOSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any {
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

	inners := make([]ro.Observable[int], m)

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
func fuzzHOBuildHigher(kind int, inners []ro.Observable[int], outer ro.Observable[int]) (obs ro.Observable[int], ordered bool) {
	project := func(v int) ro.Observable[int] { return inners[v] }

	switch kind {
	case fuzzHOKindMergeAll:
		return ro.MergeAll[int]()(ro.Map(project)(outer)), false
	case fuzzHOKindMergeMap:
		return ro.MergeMap(project)(outer), false
	case fuzzHOKindConcatAll:
		return ro.ConcatAll[int]()(ro.Map(project)(outer)), true
	case fuzzHOKindConcatWith:
		return ro.ConcatWith(inners[1:]...)(inners[0]), true
	default:
		return ro.FlatMap(project)(outer), true
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
func fuzzHOBuildLoop(kind int, src ro.Observable[int], condition func() bool, bounded bool, rounds, retryMax int, retryReset bool) ro.Observable[int] {
	switch kind {
	case fuzzHOLoopWhile:
		return ro.While[int](condition)(src)
	case fuzzHOLoopDoWhile:
		return ro.DoWhile[int](condition)(src)
	case fuzzHOLoopRepeatWith:
		count := int64(1000) // finite, so that a missing stop is slow instead of infinite.
		if bounded {
			count = int64(rounds)
		}

		return ro.RepeatWith[int](count)(src)
	case fuzzHOLoopRetry:
		return ro.Retry[int]()(src)
	default:
		return ro.RetryWithConfig[int](ro.RetryConfig{MaxRetries: uint64(retryMax), ResetOnSuccess: retryReset})(src) //nolint:gosec // retryMax is in [1,3].
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

func (s *fuzzBOSink[T]) observer() ro.Observer[T] {
	return ro.NewObserverWithContext(
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
	build func() ro.Observable[T], check func(got []T) error, counters ...*activeCounter,
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
func fuzzBOSrc(c *activeCounter, seed int64, n int, async bool) ro.Observable[int] {
	return trackSubscriptions(c, fuzzSource(seed, n, async))
}

const (
	// fuzzTimeMaxMicros keeps every delay at or below 2ms so that one iteration stays fast.
	fuzzTimeMaxMicros = 2000

	// fuzzTimeSettle gives goroutines that outlive an unsubscription the time to misbehave
	// before the post-conditions are read.
	fuzzTimeSettle = 10 * time.Millisecond

	// Downstream stop strategies.
	fuzzTimeModeComplete = 0 // run to termination (finite sources only)
	fuzzTimeModeTake     = 1 // Take(k) cancels from inside Next
	fuzzTimeModeUnsub    = 2 // external, concurrent Unsubscribe
	fuzzTimeModeCancel   = 3 // context cancellation
	fuzzTimeModeCount    = 4
)

// fuzzTimePick derives a bounded, seed-dependent value; salt keeps independent knobs decorrelated.
func fuzzTimePick(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

func fuzzTimeMicros(seed int64, salt, lo int) time.Duration {
	return time.Duration(fuzzTimePick(seed, salt, lo, fuzzTimeMaxMicros)) * time.Microsecond
}

// fuzzTimeIter runs body with panic recovery and a deadline, so a hang or a panic fails the
// iteration with the operator's name instead of crashing or blocking the whole run.
func fuzzTimeIter(t *testing.T, name string, body func() error) {
	t.Helper()

	done := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- fmt.Errorf("%s: panic: %v\n%s", name, r, debug.Stack())
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

// fuzzTimeSink is a downstream observer that records protocol violations.
type fuzzTimeSink[T any] struct {
	guard    serialGuard
	nexts    int32
	errs     int32
	comps    int32
	terminal int32
	afterEnd int32
}

func (s *fuzzTimeSink[T]) observer(slow time.Duration) ro.Observer[T] {
	return ro.NewObserverWithContext(
		func(_ context.Context, _ T) {
			s.guard.enter()
			defer s.guard.leave()

			if atomic.LoadInt32(&s.terminal) != 0 {
				atomic.AddInt32(&s.afterEnd, 1)
			}

			atomic.AddInt32(&s.nexts, 1)

			if slow > 0 {
				time.Sleep(slow)
			}
		},
		func(_ context.Context, _ error) {
			s.guard.enter()
			defer s.guard.leave()

			atomic.AddInt32(&s.terminal, 1)
			atomic.AddInt32(&s.errs, 1)
		},
		func(_ context.Context) {
			s.guard.enter()
			defer s.guard.leave()

			atomic.AddInt32(&s.terminal, 1)
			atomic.AddInt32(&s.comps, 1)
		},
	)
}

func (s *fuzzTimeSink[T]) violation() error {
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

// fuzzTimeDrive subscribes to obs `subs` times concurrently (distinct sinks, same operator
// instance), stops each subscription according to mode, and waits for them to close.
func fuzzTimeDrive[T any](obs ro.Observable[T], subs, mode, take int, seed int64, slow time.Duration) ([]*fuzzTimeSink[T], error) {
	if mode == fuzzTimeModeTake {
		// Take is composed by the caller through fuzzTimeMaybeTake; kept here for symmetry only.
		_ = take
	}

	sinks := make([]*fuzzTimeSink[T], subs)
	errs := make([]error, subs)

	var wg sync.WaitGroup

	for i := 0; i < subs; i++ {
		sinks[i] = &fuzzTimeSink[T]{}

		wg.Add(1)

		go func(i int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs[i] = fmt.Errorf("panic in subscription %d: %v\n%s", i, r, debug.Stack())
				}
			}()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			if mode == fuzzTimeModeCancel {
				go func() {
					time.Sleep(fuzzTimeMicros(seed, 40+i, 0))
					cancel()
				}()
			}

			sub := obs.SubscribeWithContext(ctx, sinks[i].observer(slow))

			if mode == fuzzTimeModeUnsub {
				go sub.Unsubscribe()
				fuzzJitter(seed, 50+i)
				sub.Unsubscribe()
			}

			sub.Wait()
		}(i)
	}

	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return sinks, err
		}
	}

	return sinks, nil
}

func fuzzTimeMaybeTake[T any](obs ro.Observable[T], mode, take int) ro.Observable[T] {
	if mode == fuzzTimeModeTake {
		return ro.Take[T](int64(take))(obs)
	}

	return obs
}

func fuzzTimeWaitUpstreamClosed(t *testing.T, c *activeCounter) {
	t.Helper()

	fuzzWaitFor(t, "upstream subscriptions to be released", func() bool { return c.activeCount() == 0 })
}

func fuzzTimeCheckSinks[T any](sinks []*fuzzTimeSink[T]) error {
	for _, s := range sinks {
		if err := s.violation(); err != nil {
			return err
		}
	}

	return nil
}

// fuzzTimeScenario is the decoded interleaving shared by most targets.
type fuzzTimeScenario struct {
	items int
	async bool
	mode  int
	take  int
	subs  int
	buf   int
	slow  time.Duration
	gap   time.Duration
}

func fuzzTimeDecode(seed int64, mask, k uint8, finite bool) fuzzTimeScenario {
	s := fuzzTimeScenario{
		items: fuzzTimePick(seed, 0, 1, fuzzMaxItems),
		async: fuzzIsAsync(mask, 0),
		mode:  fuzzBound(int64(k), 0, fuzzTimeModeCount-1),
		subs:  1 + int(mask>>7),
		buf:   fuzzTimePick(seed, 1, 1, 4),
	}

	if !finite && s.mode == fuzzTimeModeComplete {
		s.mode = fuzzTimeModeTake
	}

	s.take = fuzzTimePick(seed, 2, 1, s.items)

	if mask&2 != 0 {
		s.slow = fuzzTimeMicros(seed, 3, 0) / 8
	}

	if mask&4 != 0 {
		s.gap = fuzzTimeMicros(seed, 4, 0)
	}

	return s
}

func fuzzTimeSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i * 37), uint8(i)} })
}

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

func (s *fuzzSCRawSink[T]) IsClosed() bool { return false }

func (s *fuzzSCRawSink[T]) HasThrown() bool { return false }

func (s *fuzzSCRawSink[T]) IsCompleted() bool { return false }

func (s *fuzzSCRawSink[T]) snapshot() []T {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]T(nil), s.values...)
}

// fuzzSCSource emits 0..n-1 then completes (or fails when failAtEnd). Async sources emit from their
// own goroutine. checkClosed makes the source honour IsClosed/teardown; otherwise it emits everything.
func fuzzSCSource(seed int64, n int, async, checkClosed, failAtEnd bool, wg *sync.WaitGroup) ro.Observable[int] {
	finish := func(ctx context.Context, destination ro.Observer[int]) {
		if failAtEnd {
			destination.ErrorWithContext(ctx, errFuzzSCBoom)
			return
		}

		destination.CompleteWithContext(ctx)
	}

	if !async {
		return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
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

	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
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
func fuzzSCRunRaw[R any](t *testing.T, name string, seed int64, mask uint8, n int, build func(source ro.Observable[int]) ro.Observable[R]) *fuzzSCRawSink[R] {
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

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })
}

// fuzzSCScenario derives the item count and the decision index. k may be >= n, which means "no decision".
func fuzzSCScenario(seed int64) (n, k, variant int) {
	n = fuzzSCPick(seed, 0, 0, fuzzMaxItems)
	k = fuzzSCPick(seed, 1, 0, n+1)
	variant = fuzzSCPick(seed, 2, 0, 1)

	return n, k, variant
}

func fuzzSCWantFirst(n, k int) []int {
	if k < n {
		return []int{k}
	}

	return []int{}
}

// testWithTimeout panics when the test is still running after timeout, so that a hang
// is reported with the test's name instead of the generic `go test` timeout.
// https://github.com/stretchr/testify/issues/1101
func testWithTimeout(t *testing.T, timeout time.Duration) {
	t.Helper()

	testFinished := make(chan struct{})

	t.Cleanup(func() {
		close(testFinished)
	})

	line := ""
	funcName := ""

	var pc [1]uintptr
	n := runtime.Callers(2, pc[:])
	if n > 0 {
		frames := runtime.CallersFrames(pc[:])
		frame, _ := frames.Next()
		line = frame.File + ":" + strconv.Itoa(frame.Line)
		funcName = frame.Function
	}

	go func() {
		select {
		case <-testFinished:
		case <-time.After(timeout):
			if line == "" || funcName == "" {
				panic(fmt.Sprintf("Test timed out after: %v", timeout))
			}
			panic(fmt.Sprintf("%s: Test timed out after: %v\n%s", funcName, timeout, line))
		}
	}()
}
