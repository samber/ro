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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

const (
	// selfDrivenSourceBit selects a self-driven source (fuzzSource) instead of the hand-fed one.
	selfDrivenSourceBit = 5

	// asyncSourceBit selects whether the self-driven source emits from its own goroutine.
	asyncSourceBit = 6

	// secondSourceValueBase offsets the values of a second source so they never collide with the first.
	secondSourceValueBase = 1000
)

// connectableRecorder is a subscriber that records what it received and flags contract violations.
type connectableRecorder struct {
	guard         serialGuard
	mu            sync.Mutex
	values        []int
	terminals     int32
	afterTerminal int32
	onNext        func(v int)
}

func (r *connectableRecorder) observer() ro.Observer[int] {
	return ro.NewObserver(
		func(v int) {
			r.guard.enter()
			defer r.guard.leave()

			if atomic.LoadInt32(&r.terminals) > 0 {
				atomic.AddInt32(&r.afterTerminal, 1)
			}

			r.mu.Lock()
			r.values = append(r.values, v)
			r.mu.Unlock()

			if r.onNext != nil {
				r.onNext(v)
			}
		},
		func(error) {
			r.guard.enter()
			defer r.guard.leave()

			if atomic.AddInt32(&r.terminals, 1) > 1 {
				atomic.AddInt32(&r.afterTerminal, 1)
			}
		},
		func() {
			r.guard.enter()
			defer r.guard.leave()

			if atomic.AddInt32(&r.terminals, 1) > 1 {
				atomic.AddInt32(&r.afterTerminal, 1)
			}
		},
	)
}

func (r *connectableRecorder) count() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.values)
}

func (r *connectableRecorder) terminated() bool { return atomic.LoadInt32(&r.terminals) > 0 }

func (r *connectableRecorder) check(t *testing.T, name string) {
	t.Helper()

	if n := atomic.LoadInt32(&r.terminals); n > 1 {
		t.Fatalf("%s: %d terminal notifications, want <= 1", name, n)
	}

	if n := atomic.LoadInt32(&r.afterTerminal); n != 0 {
		t.Fatalf("%s: %d notifications after terminal", name, n)
	}

	if n := r.guard.overlapped(); n != 0 {
		t.Fatalf("%s: %d overlapping notifications", name, n)
	}
}

// connectableSource counts live subscriptions to a source. By default it is hand-fed: every
// subscription registers a destination, so a test can push values to all of them. With inner
// set, it wraps a self-driven source (sync or async) and emit/finish do nothing.
type connectableSource struct {
	mu      sync.Mutex
	dests   map[int]ro.Observer[int]
	nextID  int
	counter activeCounter
	inner   ro.Observable[int]
}

func (s *connectableSource) observable() ro.Observable[int] {
	if s.inner != nil {
		return trackSubscriptions(&s.counter, s.inner)
	}

	return trackSubscriptions(&s.counter, ro.NewUnsafeObservable(func(dest ro.Observer[int]) ro.Teardown {
		s.mu.Lock()
		if s.dests == nil {
			s.dests = map[int]ro.Observer[int]{}
		}
		id := s.nextID
		s.nextID++
		s.dests[id] = dest
		s.mu.Unlock()

		return func() {
			s.mu.Lock()
			delete(s.dests, id)
			s.mu.Unlock()
		}
	}))
}

func (s *connectableSource) snapshot() []ro.Observer[int] {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]ro.Observer[int], 0, len(s.dests))
	for _, d := range s.dests {
		out = append(out, d)
	}

	return out
}

func (s *connectableSource) emit(v int) {
	for _, d := range s.snapshot() {
		d.Next(v)
	}
}

func (s *connectableSource) finish(failed bool) {
	for _, d := range s.snapshot() {
		if failed {
			d.Error(context.Canceled)
		} else {
			d.Complete()
		}
	}
}

// newConnectableSource picks the hand-fed or the self-driven kind from mask. The self-driven kind
// always completes, and is synchronous or asynchronous depending on another bit of mask.
func newConnectableSource(seed int64, count int, mask uint8) *connectableSource {
	s := &connectableSource{}
	if mask&(1<<selfDrivenSourceBit) != 0 {
		s.inner = fuzzSource(seed, count, fuzzIsAsync(mask, asyncSourceBit))
	}

	return s
}

// pickConnectableValue derives a deterministic pseudo-random number in [0, mod) from the input, so that
// every goroutine takes different decisions per seed without a shared RNG.
func pickConnectableValue(seed int64, g, step, mod int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(g+1)*1442695040888963407 + uint64(step+1)*2862933555777941757 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return int(mixed % uint64(mod)) //nolint:gosec // mod is a small positive constant.
}

// requireNoDeadlock fails the test instead of hanging when wg does not finish within fuzzDeadline.
func requireNoDeadlock(t *testing.T, what string, wg *sync.WaitGroup) {
	t.Helper()

	done := make(chan struct{})

	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(fuzzDeadline):
		t.Fatalf("deadlock: %s did not finish within %s", what, fuzzDeadline)
	}
}

// addConnectableSeeds registers seeds matching the (seed, goroutines, items, mask, cut) signature.
func addConnectableSeeds(f *testing.F) {
	f.Helper()
	xfuzz.AddSeeds(f, func(i int) []any {
		return []any{int64(i), uint8(i), uint8(i / 3), uint8(i * 7), uint8(i / 5)}
	})
}

// FuzzConnectableConcurrent races Subscribe, Connect and Unsubscribe from several goroutines
// on the same ConnectableObservable, with a hand-fed or a sync/async self-driven source.
func FuzzConnectableConcurrent(f *testing.F) {
	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		workers := fuzzBound(int64(goroutines), 1, fuzzMaxGoroutines)
		count := fuzzBound(int64(items), 0, fuzzMaxItems)
		failed := mask&1 != 0
		resetOnDisconnect := mask&2 != 0
		terminate := mask&4 != 0

		src := newConnectableSource(seed, count, mask)
		conn := ro.ConnectableWithConfig(src.observable(), ro.ConnectableConfig[int]{
			Connector:         ro.NewPublishSubject[int],
			ResetOnDisconnect: resetOnDisconnect,
		})

		recorders := make([]*connectableRecorder, workers*3)
		for i := range recorders {
			recorders[i] = &connectableRecorder{}
		}

		var wg sync.WaitGroup

		wg.Add(1)

		go func() {
			defer wg.Done()

			for i := 0; i < count; i++ {
				fuzzJitter(seed, i)
				src.emit(i)
			}

			if terminate {
				src.finish(failed)
			}
		}()

		for g := 0; g < workers; g++ {
			wg.Add(1)

			go func(g int) {
				defer wg.Done()

				var conns []ro.Subscription

				var subs []ro.Subscription

				for step := 0; step < 3; step++ {
					fuzzJitter(seed, g*10+step)

					switch pickConnectableValue(seed+int64(cut), g, step, 4) {
					case 0:
						subs = append(subs, conn.Subscribe(recorders[g*3+step].observer()))
					case 1, 2:
						conns = append(conns, conn.Connect())
					default:
						if len(conns) > 0 {
							conns[len(conns)-1].Unsubscribe()
						} else if len(subs) > 0 {
							subs[len(subs)-1].Unsubscribe()
						}
					}
				}
			}(g)
		}

		requireNoDeadlock(t, "concurrent Connect/Subscribe/Unsubscribe", &wg)

		for _, r := range recorders {
			r.check(t, "subscriber")
		}
	})
}

// FuzzConnectableSyncReconnect connects concurrently, possibly after the source already
// completed. With ResetOnDisconnect every connection must feed a fresh subject. The source
// is synchronous or asynchronous depending on the mask.
func FuzzConnectableSyncReconnect(f *testing.F) {
	f.Skip("race: connectable-stale-subject-after-sync-complete; remove when fixed")

	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		workers := fuzzBound(int64(goroutines), 1, fuzzMaxGoroutines)
		count := fuzzBound(int64(items), 0, fuzzMaxItems)
		connectFirst := mask&1 != 0
		async := fuzzIsAsync(mask, asyncSourceBit)

		conn := ro.ConnectableWithConfig(fuzzSource(seed, count, async), ro.ConnectableConfig[int]{
			Connector:         ro.NewPublishSubject[int],
			ResetOnDisconnect: true,
		})

		recorders := make([]*connectableRecorder, workers)

		var wg sync.WaitGroup

		for g := 0; g < workers; g++ {
			recorders[g] = &connectableRecorder{}

			wg.Add(1)

			go func(g int) {
				defer wg.Done()

				fuzzJitter(seed, g)

				if connectFirst && pickConnectableValue(seed, g, 0, 2) == 0 {
					conn.Connect()
				}

				conn.Subscribe(recorders[g].observer())

				fuzzJitter(seed, g+100+int(cut))
				conn.Connect()
			}(g)
		}

		requireNoDeadlock(t, "concurrent reconnect", &wg)

		for g, r := range recorders {
			r.check(t, "subscriber")

			// With a synchronous source, a terminal without the whole sequence means the subscriber
			// was attached to a subject that had already completed, i.e. the connection did not
			// get a fresh one. An asynchronous source may legitimately be joined mid-stream.
			if !async && r.terminated() && r.count() != count {
				t.Fatalf("worker %d: completed with %d values, want %d (connection fed a stale subject)", g, r.count(), count)
			}
		}

		if async {
			return
		}

		// Quiescent: one more connection must deliver the whole sequence to a new subscriber.
		last := &connectableRecorder{}
		conn.Subscribe(last.observer())
		conn.Connect()

		if last.count() != count || !last.terminated() {
			t.Fatalf("final connection delivered %d/%d values, terminated=%v", last.count(), count, last.terminated())
		}
	})
}

// FuzzConnectableReentrant subscribes or connects from inside a subscriber callback while the
// source is emitting, for a synchronous or asynchronous source.
func FuzzConnectableReentrant(f *testing.F) {
	f.Skip("race: connectable-reentrant-deadlock (Connect holds mu across sync emission; async Subscribe-in-callback hangs the subject); remove when fixed")

	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		count := fuzzBound(int64(items), 1, fuzzMaxItems)
		trigger := fuzzBound(int64(cut), 0, count-1)
		connectInside := mask&1 != 0
		resetOnDisconnect := mask&2 != 0

		conn := ro.ConnectableWithConfig(fuzzSource(seed, count, fuzzIsAsync(mask, asyncSourceBit)), ro.ConnectableConfig[int]{
			Connector:         ro.NewPublishSubject[int],
			ResetOnDisconnect: resetOnDisconnect,
		})

		inner := &connectableRecorder{}
		outer := &connectableRecorder{}

		var fired int32

		outer.onNext = func(v int) {
			if v != trigger || !atomic.CompareAndSwapInt32(&fired, 0, 1) {
				return
			}

			fuzzJitter(seed, v)

			if connectInside {
				conn.Connect()
			} else {
				conn.Subscribe(inner.observer())
			}
		}

		var wg sync.WaitGroup

		wg.Add(1)

		go func() {
			defer wg.Done()

			conn.Subscribe(outer.observer())
			conn.Connect()
		}()

		scenario := fmt.Sprintf("connectInside=%v async=%v resetOnDisconnect=%v", connectInside, fuzzIsAsync(mask, asyncSourceBit), resetOnDisconnect)

		requireNoDeadlock(t, "re-entrant Connect/Subscribe from a callback ("+scenario+")", &wg)
		fuzzWaitFor(t, "outer subscriber to terminate ("+scenario+")", outer.terminated)

		outer.check(t, "outer")
		inner.check(t, "inner")
	})
}

// FuzzConnectableShareReset races subscribers against a source that completes or fails, for
// every combination of the Share reset flags.
func FuzzConnectableShareReset(f *testing.F) {
	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		workers := fuzzBound(int64(goroutines), 1, fuzzMaxGoroutines)
		count := fuzzBound(int64(items), 0, fuzzMaxItems)
		resetOnError := mask&1 != 0
		resetOnComplete := mask&2 != 0
		resetOnRefCountZero := mask&4 != 0
		terminate := mask&8 != 0
		failed := mask&16 != 0

		src := newConnectableSource(seed, count, mask)
		if src.inner != nil {
			// A self-driven source always completes.
			terminate, failed = true, false
		}

		shared := ro.ShareWithConfig(ro.ShareConfig[int]{
			Connector:           ro.NewPublishSubject[int],
			ResetOnError:        resetOnError,
			ResetOnComplete:     resetOnComplete,
			ResetOnRefCountZero: resetOnRefCountZero,
		})(src.observable())

		var wg sync.WaitGroup

		wg.Add(1)

		go func() {
			defer wg.Done()

			for i := 0; i < count; i++ {
				fuzzJitter(seed, i)
				src.emit(i)
			}

			if terminate {
				src.finish(failed)
			}
		}()

		recorders := make([]*connectableRecorder, workers*2)
		subs := make([]ro.Subscription, workers*2)

		var subsMu sync.Mutex

		for g := 0; g < workers; g++ {
			wg.Add(1)

			go func(g int) {
				defer wg.Done()

				for round := 0; round < 2; round++ {
					fuzzJitter(seed, g*4+round)

					rec := &connectableRecorder{}
					sub := shared.Subscribe(rec.observer())

					subsMu.Lock()
					recorders[g*2+round] = rec
					subs[g*2+round] = sub
					subsMu.Unlock()

					// Leaving early drops the refcount to zero while the source may still be emitting.
					if pickConnectableValue(seed+int64(cut), g, round, 3) == 0 {
						sub.Unsubscribe()
					}
				}
			}(g)
		}

		requireNoDeadlock(t, "concurrent Share subscribers", &wg)

		for _, r := range recorders {
			r.check(t, "subscriber")
		}

		for _, sub := range subs {
			sub.Unsubscribe()
		}

		// A reset generation has no owner left to unsubscribe its source, so the operator must
		// have done it; only the generations that were meant to be reset are asserted.
		// Without ResetOnRefCountZero, a subscriber arriving after the source ended opens a new
		// generation that legitimately stays subscribed, so nothing can be asserted.
		keptTerminal := terminate && ((failed && !resetOnError) || (!failed && !resetOnComplete))

		if !resetOnRefCountZero || keptTerminal {
			return
		}

		// An asynchronous source may still be emitting when the last subscriber leaves.
		scenario := fmt.Sprintf("resetOnError=%v resetOnComplete=%v resetOnRefCountZero=%v terminate=%v failed=%v selfDriven=%v async=%v",
			resetOnError, resetOnComplete, resetOnRefCountZero, terminate, failed, src.inner != nil, fuzzIsAsync(mask, asyncSourceBit))

		fuzzWaitFor(t, "source subscriptions to be released ("+scenario+")", func() bool { return src.counter.activeCount() == 0 })

		checkFreshGeneration(t, shared, src, count)
	})
}

// checkFreshGeneration subscribes once more to a reset Share and asserts that it opened
// exactly one fresh source subscription and received the fresh source's values.
func checkFreshGeneration(t *testing.T, shared ro.Observable[int], src *connectableSource, count int) {
	t.Helper()

	before := src.counter.totalCount()
	fresh := &connectableRecorder{}
	sub := shared.Subscribe(fresh.observer())

	defer sub.Unsubscribe()

	if got := src.counter.totalCount(); got != before+1 {
		t.Fatalf("next subscriber opened %d source subscriptions, want 1 fresh one", got-before)
	}

	if src.inner != nil {
		fuzzWaitFor(t, "fresh subscriber to terminate", fresh.terminated)

		if fresh.count() != count {
			t.Fatalf("fresh subscriber received %d values from the fresh source, want %d", fresh.count(), count)
		}

		return
	}

	src.emit(42)

	if fresh.count() != 1 {
		t.Fatalf("fresh subscriber received %d values from the fresh source, want 1", fresh.count())
	}
}

// FuzzConnectableShareReplayRefCount checks the documented contract of ShareReplay: the source is
// unsubscribed when every subscriber left.
func FuzzConnectableShareReplayRefCount(f *testing.F) {
	f.Skip("race: sharereplay-source-stays-subscribed; remove when fixed")

	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		workers := fuzzBound(int64(goroutines), 1, fuzzMaxGoroutines)
		count := fuzzBound(int64(items), 0, fuzzMaxItems)

		src := newConnectableSource(seed, count, mask)
		shared := ro.ShareReplayWithConfig[int](fuzzBound(int64(cut), 1, 8), ro.ShareReplayConfig{ResetOnRefCountZero: false})(src.observable())

		var wg sync.WaitGroup

		recorders := make([]*connectableRecorder, workers)

		for g := 0; g < workers; g++ {
			recorders[g] = &connectableRecorder{}

			wg.Add(1)

			go func(g int) {
				defer wg.Done()

				fuzzJitter(seed, g)

				sub := shared.Subscribe(recorders[g].observer())

				for i := 0; i < count && g == 0; i++ {
					src.emit(i)
				}

				fuzzJitter(seed, g+50)
				sub.Unsubscribe()
			}(g)
		}

		requireNoDeadlock(t, "ShareReplay subscribers", &wg)

		for _, r := range recorders {
			r.check(t, "subscriber")
		}

		if n := src.counter.activeCount(); n != 0 {
			t.Fatalf("ShareReplay left %d source subscriptions active after all subscribers left", n)
		}
	})
}

// checkSubscriberIsolation asserts that even-indexed recorders only saw source A values and
// odd-indexed ones only source B values.
func checkSubscriberIsolation(t *testing.T, recorders []*connectableRecorder, selfDriven bool, count int) {
	t.Helper()

	for idx, r := range recorders {
		r.check(t, "subscriber")

		// Hand-fed subscribers were registered before any emission, so each sees the whole
		// sequence; a self-driven source may start before the later subscribers join.
		if !selfDriven && r.count() != count {
			t.Fatalf("subscriber %d received %d values, want %d", idx, r.count(), count)
		}

		r.mu.Lock()
		for _, v := range r.values {
			if (idx%2 == 0) != (v < secondSourceValueBase) {
				r.mu.Unlock()
				t.Fatalf("subscriber %d (source %d) received foreign value %d", idx, idx%2, v)
			}
		}
		r.mu.Unlock()
	}
}

// FuzzConnectableShareIsolation applies one Share operator to two different sources and races
// subscribers on both: state must not leak between the two applications. Sources are
// hand-fed or self-driven (sync or async) depending on the mask.
func FuzzConnectableShareIsolation(f *testing.F) {
	addConnectableSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, goroutines, items, mask, cut uint8) {
		workers := fuzzBound(int64(goroutines), 1, fuzzMaxGoroutines)
		count := fuzzBound(int64(items), 1, fuzzMaxItems)

		op := ro.Share[int]()
		srcA, srcB := newConnectableSource(seed, count, mask), newConnectableSource(seed+1, count, mask)
		obsA := srcA.observable()
		obsB := ro.Map(func(v int) int { return v + secondSourceValueBase })(srcB.observable())
		sharedA, sharedB := op(obsA), op(obsB)
		selfDriven := srcA.inner != nil

		recorders := make([]*connectableRecorder, workers*2)
		subs := make([]ro.Subscription, workers*2)

		var wg sync.WaitGroup

		for g := 0; g < workers; g++ {
			for side := 0; side < 2; side++ {
				idx := g*2 + side
				recorders[idx] = &connectableRecorder{}

				wg.Add(1)

				go func(idx, g, side int) {
					defer wg.Done()

					fuzzJitter(seed+int64(cut), g*2+side)

					if side == 0 {
						subs[idx] = sharedA.Subscribe(recorders[idx].observer())
					} else {
						subs[idx] = sharedB.Subscribe(recorders[idx].observer())
					}
				}(idx, g, side)
			}
		}

		requireNoDeadlock(t, "subscribing to both shares", &wg)

		wg.Add(2)

		go func() {
			defer wg.Done()

			for i := 0; i < count; i++ {
				fuzzJitter(seed, i)
				srcA.emit(i)
			}
		}()

		go func() {
			defer wg.Done()

			for i := 0; i < count; i++ {
				fuzzJitter(seed, i+500)
				srcB.emit(i)
			}
		}()

		requireNoDeadlock(t, "emitting", &wg)

		checkSubscriberIsolation(t, recorders, selfDriven, count)

		for _, sub := range subs {
			sub.Unsubscribe()
		}

		fuzzWaitFor(t, "sources to be released", func() bool {
			return srcA.counter.activeCount() == 0 && srcB.counter.activeCount() == 0
		})

		if !selfDriven {
			if a, b := srcA.counter.totalCount(), srcB.counter.totalCount(); a != 1 || b != 1 {
				t.Fatalf("each source must be subscribed exactly once, got A=%d B=%d", a, b)
			}
		}
	})
}

func FuzzConnectableObservableConcurrentConnectSubscribe(f *testing.F) {
	const goroutines = 16
	// Each seed runs a short burst per goroutine; many seeds replace the former long loop.
	const maxBurst = 4

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i)} })

	f.Fuzz(func(t *testing.T, seed int64) {
		burst := fuzzBound(seed, 1, maxBurst)

		connectable := ro.Connectable(ro.Just(1, 2, 3))

		var wg sync.WaitGroup
		for i := 0; i < goroutines; i++ {
			i := i
			wg.Add(1)
			go func() {
				defer wg.Done()

				for j := 0; j < burst; j++ {
					// Connect completes synchronously and its teardown resets the subject,
					// racing with concurrent Connect and Subscribe calls.
					fuzzJitter(seed, i)
					connectable.Connect().Unsubscribe()
					fuzzJitter(seed, i+goroutines)
					connectable.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
				}
			}()
		}
		wg.Wait()
	})
}
