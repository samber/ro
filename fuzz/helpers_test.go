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
	// earlyStopNone lets the stream run to its natural end.
	earlyStopNone = 0
	// earlyStopTake closes the downstream from inside the pipeline with Take.
	earlyStopTake = 1
	// earlyStopExternal unsubscribes from outside after a few items.
	earlyStopExternal = 2

	// maxEarlyStopCount bounds the early-stop item count, so loop operators stay cheap.
	maxEarlyStopCount = 12

	// sourceTagStride separates the values of two sources: source i emits i*stride+j, j < stride.
	sourceTagStride = 1000

	// unboundedLoopCount is a loop bound that only a downstream stop can end.
	unboundedLoopCount = 1 << 30

	// boundedLoopBit selects, in the loop targets, a finite loop instead of an early stop.
	boundedLoopBit = 0x80

	// maxConcatWithArity bounds the arity of ConcatWith, which is variadic.
	maxConcatWithArity = 8

	// maxLoopRounds is the resubscription count of the finite loop variants.
	maxLoopRounds = 3
)

var errHigherOrderBoom = errors.New("higher-order: boom")

type sourceEnd int

const (
	sourceEndComplete sourceEnd = iota
	sourceEndError
	// sourceEndNever emits its items and then stays subscribed until unsubscribed.
	sourceEndNever
)

// taggedSource emits tag*stride+0..n-1 then ends as requested. A synchronous source emits inside
// Subscribe, an asynchronous one from its own goroutine.
func taggedSource(seed int64, tag, n int, async bool, end sourceEnd) ro.Observable[int] {
	finish := func(ctx context.Context, destination ro.Observer[int]) {
		switch end {
		case sourceEndComplete:
			destination.CompleteWithContext(ctx)
		case sourceEndError:
			destination.ErrorWithContext(ctx, errHigherOrderBoom)
		case sourceEndNever:
		}
	}

	if !async {
		return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
			for i := 0; i < n && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, tag*sourceTagStride+i)
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
				destination.NextWithContext(ctx, tag*sourceTagStride+i)
			}

			finish(ctx, destination)
		}()

		return func() { close(done) }
	})
}

// streamRecorder records what the downstream observes and checks the observer contract.
type streamRecorder struct {
	guard serialGuard

	nexts     int32
	errs      int32
	completes int32
	afterTerm int32

	mu     sync.Mutex
	values []int
}

func (r *streamRecorder) terminals() int {
	return int(atomic.LoadInt32(&r.errs) + atomic.LoadInt32(&r.completes))
}

func (r *streamRecorder) nextCount() int { return int(atomic.LoadInt32(&r.nexts)) }

func (r *streamRecorder) errCount() int { return int(atomic.LoadInt32(&r.errs)) }

func (r *streamRecorder) completeCount() int { return int(atomic.LoadInt32(&r.completes)) }

func (r *streamRecorder) snapshot() []int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]int(nil), r.values...)
}

func (r *streamRecorder) observer() ro.Observer[int] {
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

// streamHarness subscribes in its own goroutine so that a Subscribe that never returns is
// reported as a failure instead of hanging the test. On failure it cancels the subscriber context
// and raises stopped, which loops use as an exit, so a buggy infinite loop does not keep spinning.
type streamHarness struct {
	t       *testing.T
	ctx     context.Context
	cancel  context.CancelFunc
	stopped int32
	rec     *streamRecorder

	mu       sync.Mutex
	sub      ro.Subscription
	returned chan struct{}
}

func (h *streamHarness) isStopped() bool { return atomic.LoadInt32(&h.stopped) == 1 }

func (h *streamHarness) fail(format string, args ...any) {
	h.t.Helper()
	atomic.StoreInt32(&h.stopped, 1)
	h.cancel()
	h.t.Fatalf(format, args...)
}

func (h *streamHarness) waitFor(what string, cond func() bool) {
	h.t.Helper()

	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			h.fail("timeout waiting for: %s", what)
		}

		time.Sleep(time.Millisecond)
	}
}

func (h *streamHarness) start(obs ro.Observable[int]) {
	go func() {
		s := obs.SubscribeWithContext(h.ctx, h.rec.observer())

		h.mu.Lock()
		h.sub = s
		h.mu.Unlock()

		close(h.returned)
	}()
}

func (h *streamHarness) subscription() ro.Subscription {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.sub
}

func (h *streamHarness) hasReturned() bool {
	select {
	case <-h.returned:
		return true
	default:
		return false
	}
}

// settle drives the stream to its end according to mode, then unsubscribes.
func (h *streamHarness) settle(mode, kk int) {
	h.t.Helper()

	if mode == earlyStopExternal {
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
func (h *streamHarness) verify(counters ...*activeCounter) {
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

func newStreamHarness(t *testing.T) *streamHarness {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	h := &streamHarness{t: t, ctx: ctx, cancel: cancel, rec: &streamRecorder{}, returned: make(chan struct{})}

	t.Cleanup(func() {
		atomic.StoreInt32(&h.stopped, 1)
		cancel()
	})

	return h
}

// decodeEarlyStop derives the early-stop mode and item count from one fuzz input.
func decodeEarlyStop(k int64) (mode, kk int) {
	return fuzzBound(k, earlyStopNone, earlyStopExternal), fuzzBound(k/3, 1, maxEarlyStopCount)
}

func applyEarlyStop(obs ro.Observable[int], mode, kk int) ro.Observable[int] {
	if mode == earlyStopTake {
		return ro.Take[int](int64(kk))(obs)
	}

	return obs
}

func addStreamSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any {
		return []any{int64(i), int64(i*3 + 1), uint8(i*37 + i/7), int64(i*5 + 2)} //nolint:gosec // wrap-around is intended.
	})
}

func minInt(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// ---------------------------------------------------------------------------------------------
// MergeAll / MergeMap / ConcatAll / ConcatWith / FlatMap
// ---------------------------------------------------------------------------------------------

const (
	higherOrderKindMergeAll = iota
	higherOrderKindMergeMap
	higherOrderKindConcatAll
	higherOrderKindConcatWith
	higherOrderKindFlatMap
)

func runHigherOrder(t *testing.T, kind int, seed, size int64, mask uint8, k int64) {
	t.Helper()

	h := newStreamHarness(t)
	mode, kk := decodeEarlyStop(k)

	m := fuzzBound(size, 1, fuzzMaxItems)
	if kind == higherOrderKindConcatWith {
		m = fuzzBound(size, 1, maxConcatWithArity)
	}

	var innerC, outerC activeCounter

	inners := make([]ro.Observable[int], m)

	var expected []int

	for i := range inners {
		n := fuzzBound(seed+int64(i)*7, 0, 4)
		inners[i] = trackSubscriptions(&innerC, taggedSource(seed+int64(i), i, n, fuzzIsAsync(mask, i+1), sourceEndComplete))

		for j := 0; j < n; j++ {
			expected = append(expected, i*sourceTagStride+j)
		}
	}

	outer := trackSubscriptions(&outerC, fuzzSource(seed, m, fuzzIsAsync(mask, 0)))
	obs, ordered := buildHigherOrder(kind, inners, outer)

	h.start(applyEarlyStop(obs, mode, kk))
	h.settle(mode, kk)
	h.verify(&innerC, &outerC)

	checkHigherOrder(h, mode, kk, expected, ordered)
}

// buildHigherOrder builds the higher-order operator under test. ordered reports whether the
// operator preserves the order of the inner sources.
func buildHigherOrder(kind int, inners []ro.Observable[int], outer ro.Observable[int]) (obs ro.Observable[int], ordered bool) {
	project := func(v int) ro.Observable[int] { return inners[v] }

	switch kind {
	case higherOrderKindMergeAll:
		return ro.MergeAll[int]()(ro.Map(project)(outer)), false
	case higherOrderKindMergeMap:
		return ro.MergeMap(project)(outer), false
	case higherOrderKindConcatAll:
		return ro.ConcatAll[int]()(ro.Map(project)(outer)), true
	case higherOrderKindConcatWith:
		return ro.ConcatWith(inners[1:]...)(inners[0]), true
	default:
		return ro.FlatMap(project)(outer), true
	}
}

// checkHigherOrder asserts the values and terminal notifications delivered by a higher-order
// operator against the values the inner sources can emit.
func checkHigherOrder(h *streamHarness, mode, kk int, expected []int, ordered bool) {
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
	case earlyStopNone:
		if len(got) != len(expected) || h.rec.errCount() != 0 || h.rec.completeCount() != 1 {
			h.fail("lost values or terminal: got %d/%d values, errs=%d completes=%d", len(got), len(expected), h.rec.errCount(), h.rec.completeCount())
		}
	case earlyStopTake:
		if len(got) != minInt(kk, len(expected)) {
			h.fail("take(%d) delivered %d of %d available values", kk, len(got), len(expected))
		}
	}
}

// ---------------------------------------------------------------------------------------------
// Loops: While / DoWhile / RepeatWith / Retry / RetryWithConfig, and OnErrorResumeNextWith
// ---------------------------------------------------------------------------------------------

const (
	loopKindWhile = iota
	loopKindDoWhile
	loopKindRepeatWith
	loopKindRetry
	loopKindRetryWithConfig
)

func runLoop(t *testing.T, kind int, seed, size int64, mask uint8, k int64) {
	t.Helper()

	h := newStreamHarness(t)
	n := fuzzBound(size, 1, 4)
	kk := fuzzBound(k, 1, maxEarlyStopCount)
	bounded := mask&boundedLoopBit != 0 && kind <= loopKindRepeatWith
	rounds := fuzzBound(k/7, 1, maxLoopRounds)

	end := sourceEndComplete
	if kind >= loopKindRetry {
		end = sourceEndError
	}

	var srcC activeCounter

	src := trackSubscriptions(&srcC, taggedSource(seed, 0, n, fuzzIsAsync(mask, 0), end))

	// Subscriptions needed to deliver kk items: the loop must not open another one afterwards.
	need := (kk + n - 1) / n

	// condition is evaluated once per loop iteration. The stop flag only bounds a buggy loop.
	var calls int32

	limit := int32(unboundedLoopCount)
	if bounded {
		limit = int32(rounds) //nolint:gosec // rounds is in [1,3].
	}

	condition := func() bool { return !h.isStopped() && atomic.AddInt32(&calls, 1) <= limit }

	retryMax := fuzzBound(k/7, 1, maxLoopRounds)
	retryReset := mask&0x40 != 0

	obs := buildLoop(kind, src, condition, bounded, rounds, retryMax, retryReset)

	mode := earlyStopTake
	if bounded {
		mode = earlyStopNone
	}

	h.start(applyEarlyStop(obs, mode, kk))
	h.settle(mode, kk)
	h.verify(&srcC)

	checkLoop(h, kind, srcC.totalCount(), n, kk, need, rounds, retryMax, retryReset, bounded)
}

// buildLoop builds the looping operator under test around src.
func buildLoop(kind int, src ro.Observable[int], condition func() bool, bounded bool, rounds, retryMax int, retryReset bool) ro.Observable[int] {
	switch kind {
	case loopKindWhile:
		return ro.While[int](condition)(src)
	case loopKindDoWhile:
		return ro.DoWhile[int](condition)(src)
	case loopKindRepeatWith:
		count := int64(1000) // finite, so that a missing stop is slow instead of infinite.
		if bounded {
			count = int64(rounds)
		}

		return ro.RepeatWith[int](count)(src)
	case loopKindRetry:
		return ro.Retry[int]()(src)
	default:
		return ro.RetryWithConfig[int](ro.RetryConfig{MaxRetries: uint64(retryMax), ResetOnSuccess: retryReset})(src) //nolint:gosec // retryMax is in [1,3].
	}
}

// checkLoop asserts the number of source subscriptions and delivered values of a loop
// operator: n is the item count per run, kk the take size, need the subscriptions required to
// deliver kk items.
func checkLoop(h *streamHarness, kind, subs, n, kk, need, rounds, retryMax int, retryReset, bounded bool) {
	got := h.rec.nextCount()

	if bounded {
		wantSubs := rounds
		if kind == loopKindDoWhile {
			wantSubs = rounds + 1 // the first run is unconditional, then one per true condition.
		}

		if subs != wantSubs || got != wantSubs*n || h.rec.completeCount() != 1 {
			h.fail("bounded loop: %d subscriptions (want %d), %d values (want %d), completes=%d", subs, wantSubs, got, wantSubs*n, h.rec.completeCount())
		}

		return
	}

	if kind == loopKindRetryWithConfig && !retryReset && kk > (retryMax+1)*n {
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
	// maxBoundaryMicros keeps every timer at or below 2ms so that one iteration stays fast.
	maxBoundaryMicros = 2000

	// maxUnsubscribeSpins bounds the busy-yield loop that delays an external Unsubscribe.
	maxUnsubscribeSpins = 24

	// boundarySettleDelay lets goroutines that outlive an unsubscription misbehave before post-conditions are read.
	boundarySettleDelay = 3 * time.Millisecond

	// unsubscribeBit selects the "unsubscribe mid-stream" scenario in the fuzz mask.
	unsubscribeBit = 0x80

	// maxBoundaryTicks bounds the number of boundary ticks a fuzz input can request.
	maxBoundaryTicks = 12
)

// pickBoundaryValue derives a bounded, seed-dependent value; salt decorrelates independent knobs.
func pickBoundaryValue(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

// waitForCondition polls cond until true or fuzzDeadline. Unlike fuzzWaitFor it returns an error, because
// it runs outside the test goroutine where t.Fatalf would only kill the helper goroutine.
func waitForCondition(what string, cond func() bool) error {
	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			return fmt.Errorf("timeout waiting for: %s", what)
		}

		time.Sleep(200 * time.Microsecond)
	}

	return nil
}

// runBoundaryIteration runs body with panic recovery and a deadline so a hang or panic fails the iteration.
func runBoundaryIteration(t *testing.T, name string, body func() error) {
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

// boundarySink records everything a downstream observer receives.
type boundarySink[T any] struct {
	mu       sync.Mutex
	vals     []T
	guard    serialGuard
	terminal int32
	errs     int32
	afterEnd int32
	lastErr  atomic.Value
}

func (s *boundarySink[T]) observer() ro.Observer[T] {
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

func (s *boundarySink[T]) snapshot() []T {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]T(nil), s.vals...)
}

func (s *boundarySink[T]) done() bool { return atomic.LoadInt32(&s.terminal) > 0 }

func (s *boundarySink[T]) violation() error {
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
func (s *boundarySink[T]) unexpectedErr() error {
	if atomic.LoadInt32(&s.errs) > 0 {
		return fmt.Errorf("unexpected error notification: %v", s.lastErr.Load())
	}

	return nil
}

func activeSubscriptionTotal(counters []*activeCounter) int {
	total := 0
	for _, c := range counters {
		total += c.activeCount()
	}

	return total
}

// runBoundaryOperator subscribes, then either waits for completion or unsubscribes after a seed-dependent
// delay. In both cases every tracked upstream/boundary subscription must be released.
// check, when non-nil, validates the received values after a normal completion.
func runBoundaryOperator[T any](
	t *testing.T, name string, seed int64, unsub bool,
	build func() ro.Observable[T], check func(got []T) error, counters ...*activeCounter,
) {
	t.Helper()

	runBoundaryIteration(t, name, func() error {
		sink := &boundarySink[T]{}
		sub := build().Subscribe(sink.observer())

		if unsub {
			for i, n := 0, pickBoundaryValue(seed, 101, 0, maxUnsubscribeSpins); i < n; i++ {
				fuzzJitter(seed, i)
			}

			sub.Unsubscribe()
		} else if err := waitForCondition("downstream terminal notification", sink.done); err != nil {
			return err
		}

		if err := waitForCondition("upstream and boundary subscriptions released", func() bool {
			return activeSubscriptionTotal(counters) == 0
		}); err != nil {
			return fmt.Errorf("%w (still active: %d, unsub=%v)", err, activeSubscriptionTotal(counters), unsub)
		}

		time.Sleep(boundarySettleDelay)

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

// countedSource is a finite 0..n-1 source whose subscriptions are counted.
func countedSource(c *activeCounter, seed int64, n int, async bool) ro.Observable[int] {
	return trackSubscriptions(c, fuzzSource(seed, n, async))
}

const (
	// maxTimerMicros keeps every delay at or below 2ms so that one iteration stays fast.
	maxTimerMicros = 2000

	// timerSettleDelay gives goroutines that outlive an unsubscription the time to misbehave
	// before the post-conditions are read.
	timerSettleDelay = 10 * time.Millisecond

	// Downstream stop strategies.
	timerStopComplete    = 0 // run to termination (finite sources only)
	timerStopTake        = 1 // Take(k) cancels from inside Next
	timerStopUnsubscribe = 2 // external, concurrent Unsubscribe
	timerStopCancel      = 3 // context cancellation
	timerStopModeCount   = 4
)

// pickTimerValue derives a bounded, seed-dependent value; salt keeps independent knobs decorrelated.
func pickTimerValue(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

func timerDelay(seed int64, salt, lo int) time.Duration {
	return time.Duration(pickTimerValue(seed, salt, lo, maxTimerMicros)) * time.Microsecond
}

// runTimerIteration runs body with panic recovery and a deadline, so a hang or a panic fails the
// iteration with the operator's name instead of crashing or blocking the whole run.
func runTimerIteration(t *testing.T, name string, body func() error) {
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

// timerSink is a downstream observer that records protocol violations.
type timerSink[T any] struct {
	guard    serialGuard
	nexts    int32
	errs     int32
	comps    int32
	terminal int32
	afterEnd int32
}

func (s *timerSink[T]) observer(slow time.Duration) ro.Observer[T] {
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

func (s *timerSink[T]) violation() error {
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

// driveSubscribers subscribes to obs `subs` times concurrently (distinct sinks, same operator
// instance), stops each subscription according to mode, and waits for them to close.
func driveSubscribers[T any](obs ro.Observable[T], subs, mode, take int, seed int64, slow time.Duration) ([]*timerSink[T], error) {
	if mode == timerStopTake {
		// Take is composed by the caller through applyTakeStop; kept here for symmetry only.
		_ = take
	}

	sinks := make([]*timerSink[T], subs)
	errs := make([]error, subs)

	var wg sync.WaitGroup

	for i := 0; i < subs; i++ {
		sinks[i] = &timerSink[T]{}

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

			if mode == timerStopCancel {
				go func() {
					time.Sleep(timerDelay(seed, 40+i, 0))
					cancel()
				}()
			}

			sub := obs.SubscribeWithContext(ctx, sinks[i].observer(slow))

			if mode == timerStopUnsubscribe {
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

func applyTakeStop[T any](obs ro.Observable[T], mode, take int) ro.Observable[T] {
	if mode == timerStopTake {
		return ro.Take[T](int64(take))(obs)
	}

	return obs
}

func waitUpstreamClosed(t *testing.T, c *activeCounter) {
	t.Helper()

	fuzzWaitFor(t, "upstream subscriptions to be released", func() bool { return c.activeCount() == 0 })
}

func checkTimerSinks[T any](sinks []*timerSink[T]) error {
	for _, s := range sinks {
		if err := s.violation(); err != nil {
			return err
		}
	}

	return nil
}

// timerScenario is the decoded interleaving shared by most targets.
type timerScenario struct {
	items int
	async bool
	mode  int
	take  int
	subs  int
	buf   int
	slow  time.Duration
	gap   time.Duration
}

func decodeTimerScenario(seed int64, mask, k uint8, finite bool) timerScenario {
	s := timerScenario{
		items: pickTimerValue(seed, 0, 1, fuzzMaxItems),
		async: fuzzIsAsync(mask, 0),
		mode:  fuzzBound(int64(k), 0, timerStopModeCount-1),
		subs:  1 + int(mask>>7),
		buf:   pickTimerValue(seed, 1, 1, 4),
	}

	if !finite && s.mode == timerStopComplete {
		s.mode = timerStopTake
	}

	s.take = pickTimerValue(seed, 2, 1, s.items)

	if mask&2 != 0 {
		s.slow = timerDelay(seed, 3, 0) / 8
	}

	if mask&4 != 0 {
		s.gap = timerDelay(seed, 4, 0)
	}

	return s
}

func addTimerSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i * 37), uint8(i)} })
}

var errShortCircuitBoom = errors.New("short-circuit: boom")

const (
	// shortCircuitSettleDelay lets goroutines that outlive a decision misbehave before post-conditions are read.
	shortCircuitSettleDelay = 3 * time.Millisecond

	// unreachableDefault is the fallback emitted by ElementAtOrDefault. It is outside the source range [0, n).
	unreachableDefault = -1

	// maxSchedulerYields bounds the scheduler yields before an external Unsubscribe.
	maxSchedulerYields = 50

	// Upstream stop modes.
	upstreamStopUnsubscribe = 1 // downstream unsubscribes concurrently
	upstreamStopError       = 2 // source ends with an error
	upstreamStopModeCount   = 4 // modes 0 and 3 run to a normal completion
)

// pickShortCircuitValue derives a bounded, seed-dependent value; salt decorrelates independent knobs.
func pickShortCircuitValue(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

// runShortCircuitIteration runs body with panic recovery and a deadline, so that a hang or a panic fails
// the iteration with the operator's name.
func runShortCircuitIteration(t *testing.T, name string, body func() error) {
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

// predicateCallCounter counts user predicate calls, and the calls made after the deciding one.
type predicateCallCounter struct {
	calls   int32
	decided int32
	extra   int32
}

func (p *predicateCallCounter) call(decides bool) {
	atomic.AddInt32(&p.calls, 1)

	if atomic.LoadInt32(&p.decided) != 0 {
		atomic.AddInt32(&p.extra, 1)
	}

	if decides {
		atomic.StoreInt32(&p.decided, 1)
	}
}

func (p *predicateCallCounter) extraCalls() int { return int(atomic.LoadInt32(&p.extra)) }

// rawNotificationSink is a destination that counts every notification it receives,
// including the ones sent after a terminal notification.
type rawNotificationSink[T any] struct {
	guard     serialGuard
	mu        sync.Mutex
	values    []T
	errs      int32
	comps     int32
	afterTerm int32
}

func (s *rawNotificationSink[T]) terminated() bool {
	return atomic.LoadInt32(&s.errs)+atomic.LoadInt32(&s.comps) > 0
}

func (s *rawNotificationSink[T]) Next(v T) { s.NextWithContext(context.Background(), v) }

func (s *rawNotificationSink[T]) NextWithContext(_ context.Context, v T) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	s.mu.Lock()
	s.values = append(s.values, v)
	s.mu.Unlock()
}

func (s *rawNotificationSink[T]) Error(err error) { s.ErrorWithContext(context.Background(), err) }

func (s *rawNotificationSink[T]) ErrorWithContext(_ context.Context, _ error) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	atomic.AddInt32(&s.errs, 1)
}

func (s *rawNotificationSink[T]) Complete() { s.CompleteWithContext(context.Background()) }

func (s *rawNotificationSink[T]) CompleteWithContext(_ context.Context) {
	s.guard.enter()
	defer s.guard.leave()

	if s.terminated() {
		atomic.AddInt32(&s.afterTerm, 1)
	}

	atomic.AddInt32(&s.comps, 1)
}

func (s *rawNotificationSink[T]) IsClosed() bool { return false }

func (s *rawNotificationSink[T]) HasThrown() bool { return false }

func (s *rawNotificationSink[T]) IsCompleted() bool { return false }

func (s *rawNotificationSink[T]) snapshot() []T {
	s.mu.Lock()
	defer s.mu.Unlock()

	return append([]T(nil), s.values...)
}

// shortCircuitSource emits 0..n-1 then completes (or fails when failAtEnd). Async sources emit from their
// own goroutine. checkClosed makes the source honour IsClosed/teardown; otherwise it emits everything.
func shortCircuitSource(seed int64, n int, async, checkClosed, failAtEnd bool, wg *sync.WaitGroup) ro.Observable[int] {
	finish := func(ctx context.Context, destination ro.Observer[int]) {
		if failAtEnd {
			destination.ErrorWithContext(ctx, errShortCircuitBoom)
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

// runWithRawSink subscribes build(source) with a counting destination, waits for the source to
// finish and for stray notifications to settle, then returns what the destination saw.
func runWithRawSink[R any](t *testing.T, name string, seed int64, mask uint8, n int, build func(source ro.Observable[int]) ro.Observable[R]) *rawNotificationSink[R] {
	t.Helper()

	async := fuzzIsAsync(mask, 0)
	checkClosed := fuzzIsAsync(mask, 1)

	sink := &rawNotificationSink[R]{}

	runShortCircuitIteration(t, name, func() error {
		var wg sync.WaitGroup

		obs := build(shortCircuitSource(seed, n, async, checkClosed, false, &wg))

		sub := obs.SubscribeWithContext(context.Background(), sink)

		wg.Wait()
		time.Sleep(shortCircuitSettleDelay)

		sub.Unsubscribe()

		return nil
	})

	return sink
}

// checkSingleResult asserts the single-result invariant: exactly the wanted values, exactly one terminal
// notification, nothing after it, no overlapping calls, and no predicate call after the decision.
func checkSingleResult[R any](t *testing.T, name string, sink *rawNotificationSink[R], probe *predicateCallCounter, wantValues []R, wantErr bool, mask uint8) {
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

func addShortCircuitSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })
}

// decodeShortCircuitScenario derives the item count and the decision index. k may be >= n, which means "no decision".
func decodeShortCircuitScenario(seed int64) (n, k, variant int) {
	n = pickShortCircuitValue(seed, 0, 0, fuzzMaxItems)
	k = pickShortCircuitValue(seed, 1, 0, n+1)
	variant = pickShortCircuitValue(seed, 2, 0, 1)

	return n, k, variant
}

func expectedItemAt(n, k int) []int {
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
