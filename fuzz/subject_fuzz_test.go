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
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

const (
	// subjectKindCount is the number of subject implementations a fuzz input can select.
	subjectKindCount = 5

	// subjectShortDeadline bounds waits that must NOT be blocked by a slow observer. It is
	// short on purpose: the assertion is that the call returns while another goroutine is still
	// stuck inside an observer.
	subjectShortDeadline = 300 * time.Millisecond

	// reentrantNextValue is the value pushed by a re-entrant Next. Sources only emit
	// 0..n-1 with n <= fuzzMaxItems, so it never collides with them.
	reentrantNextValue = 1 << 20

	// gapValueBase offsets values emitted while no observer is attached, so they can be
	// told apart from values emitted by the source afterwards.
	gapValueBase = 1 << 10

	// maxConcurrentProducers bounds the number of concurrent Next callers.
	maxConcurrentProducers = 4
)

var errSubjectBoom = errors.New("subject: boom")

// newSubject builds one of the 5 subject types. bufSize feeds the replay and unicast buffers
// and covers -1 (unlimited), 0 and small positive values.
func newSubject(kind, bufSize uint8) ro.Subject[int] {
	size := int(bufSize%5) - 1

	switch kind % subjectKindCount {
	case 0:
		return ro.NewPublishSubject[int]()
	case 1:
		return ro.NewBehaviorSubject(-1)
	case 2:
		return ro.NewReplaySubject[int](size)
	case 3:
		return ro.NewAsyncSubject[int]()
	default:
		return ro.NewUnicastSubject[int](size)
	}
}

// feedSubject pipes fuzzSource(0..n-1) into subject and terminates it with Complete, or Error when
// fail is set. A synchronous source emits inside Subscribe, an asynchronous one from its own goroutine.
// wg is released once the subject has been terminated.
func feedSubject(wg *sync.WaitGroup, seed int64, subject ro.Subject[int], n int, async, fail bool) {
	done := make(chan struct{})

	wg.Add(1)

	go func() {
		defer wg.Done()

		sub := fuzzSource(seed, n, async).Subscribe(ro.NewObserver(
			subject.Next,
			subject.Error,
			func() {
				if fail {
					subject.Error(errSubjectBoom)
				} else {
					subject.Complete()
				}

				close(done)
			},
		))

		<-done
		sub.Unsubscribe()
	}()
}

// subjectRecorder records what one subscriber observes and checks the observer contract.
type subjectRecorder struct {
	guard serialGuard

	nexts      int32
	errs       int32
	completes  int32
	afterTerm  int32
	afterUnsub int32
	unsubbed   int32 // set by hooks once they unsubscribed the subscriber.

	hookOnce int32
	// onEvent runs once, inside the first callback of any kind.
	onEvent func()
	// onNext runs inside every Next callback, after counting.
	onNext func(value int)

	mu     sync.Mutex
	values []int
}

func (r *subjectRecorder) terminals() int {
	return int(atomic.LoadInt32(&r.errs) + atomic.LoadInt32(&r.completes))
}

func (r *subjectRecorder) begin() {
	r.guard.enter()

	if r.terminals() > 0 {
		atomic.AddInt32(&r.afterTerm, 1)
	}

	if atomic.LoadInt32(&r.unsubbed) == 1 {
		atomic.AddInt32(&r.afterUnsub, 1)
	}

	if r.onEvent != nil && atomic.CompareAndSwapInt32(&r.hookOnce, 0, 1) {
		r.onEvent()
	}
}

func (r *subjectRecorder) observer() ro.Observer[int] {
	return ro.NewObserver(
		func(value int) {
			r.begin()
			defer r.guard.leave()

			atomic.AddInt32(&r.nexts, 1)

			r.mu.Lock()
			r.values = append(r.values, value)
			r.mu.Unlock()

			if r.onNext != nil {
				r.onNext(value)
			}
		},
		func(error) {
			r.begin()
			defer r.guard.leave()

			atomic.AddInt32(&r.errs, 1)
		},
		func() {
			r.begin()
			defer r.guard.leave()

			atomic.AddInt32(&r.completes, 1)
		},
	)
}

func (r *subjectRecorder) snapshot() []int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]int(nil), r.values...)
}

// check asserts the invariants that hold for every subscriber, whatever the interleaving.
func (r *subjectRecorder) check(t *testing.T, name string) {
	t.Helper()

	if got := r.terminals(); got > 1 {
		t.Errorf("%s: %d terminal notifications, want at most 1", name, got)
	}

	if got := atomic.LoadInt32(&r.afterTerm); got != 0 {
		t.Errorf("%s: %d notifications delivered after a terminal", name, got)
	}

	if got := r.guard.overlapped(); got != 0 {
		t.Errorf("%s: %d overlapping notifications", name, got)
	}
}

// waitGroupDoneWithin reports whether wg finished within d, so that a deadlock fails the test instead of hanging it.
func waitGroupDoneWithin(d time.Duration, wg *sync.WaitGroup) bool {
	done := make(chan struct{})

	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		return true
	case <-time.After(d):
		return false
	}
}

// requireWaitGroupDone fails the test when wg does not finish within fuzzDeadline.
func requireWaitGroupDone(t *testing.T, what string, wg *sync.WaitGroup) {
	t.Helper()

	if !waitGroupDoneWithin(fuzzDeadline, wg) {
		t.Fatalf("deadlock: %s did not finish within %s", what, fuzzDeadline)
	}
}

// requireReturnsWithin runs fn in its own goroutine and fails the test when it does not return in time.
func requireReturnsWithin(t *testing.T, what string, fn func()) {
	t.Helper()

	var wg sync.WaitGroup

	wg.Add(1)

	go func() {
		defer wg.Done()

		fn()
	}()

	requireWaitGroupDone(t, what, &wg)
}

// subscriptionHolder hands a Subscription to a callback that may run before Subscribe returns.
type subscriptionHolder struct {
	mu  sync.Mutex
	sub ro.Subscription
}

func (h *subscriptionHolder) set(sub ro.Subscription) {
	h.mu.Lock()
	h.sub = sub
	h.mu.Unlock()
}

func (h *subscriptionHolder) get() ro.Subscription {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.sub
}

func addSubjectSeeds(f *testing.F) {
	f.Helper()

	xfuzz.AddSeeds(f, func(i int) []any {
		return []any{int64(i), uint8(i), uint8(i / 5), uint8(i / 3), uint8(i * 37), uint8(i * 11), uint8(i * 13)}
	})
}

// FuzzSubjectConcurrentSubscribeNext subscribes from many goroutines while a source emits and terminates.
func FuzzSubjectConcurrentSubscribeNext(f *testing.F) {
	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := newSubject(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)

		recorders := make([]*subjectRecorder, nSubs)
		unsubscribed := make([]bool, nSubs)

		var wg sync.WaitGroup

		for i := range recorders {
			recorders[i] = &subjectRecorder{}
			unsubscribed[i] = (mask>>(uint(i)%8))&1 == 1

			wg.Add(1)

			go func(i int) {
				defer wg.Done()

				fuzzJitter(seed, i)

				sub := subject.Subscribe(recorders[i].observer())

				fuzzJitter(seed, i+100)

				if unsubscribed[i] {
					sub.Unsubscribe()
				}
			}(i)
		}

		feedSubject(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), mask&0x80 != 0)

		requireWaitGroupDone(t, "concurrent subscribe + next", &wg)

		for i, r := range recorders {
			r.check(t, "subscriber")

			// A subscriber that never unsubscribed outlives the terminal, so it must have seen exactly one.
			if !unsubscribed[i] && r.terminals() != 1 {
				t.Errorf("subscriber %d: %d terminal notifications, want exactly 1", i, r.terminals())
			}
		}
	})
}

// FuzzSubjectUnsubscribeDuringBroadcast unsubscribes from another goroutine and from inside onNext.
func FuzzSubjectUnsubscribeDuringBroadcast(f *testing.F) {
	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := newSubject(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		unsubAfter := fuzzBound(int64(extra), 0, nItems-1)

		recorders := make([]*subjectRecorder, nSubs)
		holders := make([]*subscriptionHolder, nSubs)

		for i := range recorders {
			r := &subjectRecorder{}
			h := &subscriptionHolder{}

			if (mask>>(uint(i)%8))&1 == 1 {
				r.onNext = func(int) {
					if int(atomic.LoadInt32(&r.nexts)) > unsubAfter && atomic.LoadInt32(&r.unsubbed) == 0 {
						if sub := h.get(); sub != nil {
							sub.Unsubscribe()
							atomic.StoreInt32(&r.unsubbed, 1)
						}
					}
				}
			}

			recorders[i] = r
			holders[i] = h

			requireReturnsWithin(t, "subscribe", func() {
				h.set(subject.Subscribe(r.observer()))
			})
		}

		var wg sync.WaitGroup

		// Outside unsubscriber: races with the broadcast.
		wg.Add(1)

		go func() {
			defer wg.Done()

			for i, h := range holders {
				fuzzJitter(seed, i+300)

				if (mask>>(uint(i)%8))&1 == 0 && i%2 == 0 {
					if sub := h.get(); sub != nil {
						sub.Unsubscribe()
					}
				}
			}
		}()

		feedSubject(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), false)

		requireWaitGroupDone(t, "unsubscribe during broadcast", &wg)

		for _, r := range recorders {
			r.check(t, "subscriber")

			if got := atomic.LoadInt32(&r.afterUnsub); got != 0 {
				t.Errorf("subscriber: %d notifications after it unsubscribed itself", got)
			}
		}

		if got := subject.CountObservers(); got != 0 {
			t.Errorf("%d observers still registered after completion", got)
		}
	})
}

// FuzzSubjectNextRacesTerminal races Next against Complete and Error from several goroutines.
func FuzzSubjectNextRacesTerminal(f *testing.F) {
	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := newSubject(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)
		nProducers := fuzzBound(int64(extra), 1, maxConcurrentProducers)

		recorders := make([]*subjectRecorder, nSubs)
		for i := range recorders {
			r := &subjectRecorder{}
			recorders[i] = r

			requireReturnsWithin(t, "subscribe", func() {
				subject.Subscribe(r.observer())
			})
		}

		var wg sync.WaitGroup

		// Producer 0 is a source that also terminates the subject; the others call Next directly.
		// Together with the two terminators below, up to 3 terminal calls race.
		feedSubject(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), mask&0x40 != 0)

		for p := 1; p < nProducers; p++ {
			wg.Add(1)

			go func(p int) {
				defer wg.Done()

				for i := 0; i < nItems; i++ {
					subject.Next(p*1000 + i)
					fuzzJitter(seed, i+p*7)
				}
			}(p)
		}

		for k := 0; k < 2; k++ {
			wg.Add(1)

			go func(k int) {
				defer wg.Done()

				fuzzJitter(seed, 500+k)

				if (mask>>uint(k))&1 == 1 {
					subject.Error(errSubjectBoom)
				} else {
					subject.Complete()
				}
			}(k)
		}

		requireWaitGroupDone(t, "next vs terminal", &wg)

		settled := make([]int32, nSubs)

		for i, r := range recorders {
			r.check(t, "subscriber")

			if r.terminals() != 1 {
				t.Errorf("subscriber %d: %d terminal notifications, want exactly 1", i, r.terminals())
			}

			settled[i] = atomic.LoadInt32(&r.nexts)
		}

		if !subject.IsClosed() {
			t.Errorf("subject is not closed after a terminal")
		}

		subject.Next(-42)

		for i, r := range recorders {
			if got := atomic.LoadInt32(&r.nexts); got != settled[i] {
				t.Errorf("subscriber %d received a Next after the subject terminated", i)
			}
		}
	})
}

// registerReentrantFuzz calls op from inside the first notification an observer receives.
// op: 0 = IsClosed, 1 = Next, 2 = Subscribe.
func registerReentrantFuzz(f *testing.F, op int) {
	f.Helper()
	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := newSubject(kind, extra)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)
		pre := nItems / 2

		rec := &subjectRecorder{}
		rec.onEvent = func() {
			switch op {
			case 0:
				_ = subject.IsClosed()
			case 1:
				subject.Next(reentrantNextValue)
			default:
				subject.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
			}
		}

		var wg sync.WaitGroup

		requireReturnsWithin(t, "re-entrant call", func() {
			// Items sent before Subscribe exercise the replay paths, which run the observer under Subscribe.
			for i := 0; i < pre; i++ {
				subject.Next(i)
			}

			subject.Subscribe(rec.observer())
		})

		feedSubject(&wg, seed, subject, nItems-pre, fuzzIsAsync(amask, 0), mask&1 == 1)

		requireWaitGroupDone(t, "re-entrant call", &wg)

		rec.check(t, "subscriber")
	})
}

// FuzzSubjectReentrantIsClosed calls subject.IsClosed() from inside onNext.
func FuzzSubjectReentrantIsClosed(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	registerReentrantFuzz(f, 0)
}

// FuzzSubjectReentrantNext calls subject.Next() from inside onNext.
func FuzzSubjectReentrantNext(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	registerReentrantFuzz(f, 1)
}

// FuzzSubjectReentrantSubscribe calls subject.Subscribe() from inside onNext.
func FuzzSubjectReentrantSubscribe(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	registerReentrantFuzz(f, 2)
}

// FuzzSubjectUnicastClosedSubscriber subscribes an already-closed Subscriber to a UnicastSubject.
func FuzzSubjectUnicastClosedSubscriber(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := ro.NewUnicastSubject[int](int(extra%5) - 1)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)

		for i := 0; i < nItems; i++ {
			subject.Next(i)
		}

		switch mask % 3 {
		case 1:
			subject.Complete()
		case 2:
			subject.Error(errSubjectBoom)
		}

		closed := &subjectRecorder{}
		sub := ro.NewSubscriber(closed.observer())
		sub.Unsubscribe()

		requireReturnsWithin(t, "Subscribe with a closed Subscriber", func() {
			subject.Subscribe(sub)
		})

		// The subject must stay usable afterwards.
		live := &subjectRecorder{}

		requireReturnsWithin(t, "Subscribe after a closed Subscriber", func() {
			subject.Subscribe(live.observer())
		})

		var wg sync.WaitGroup

		feedSubject(&wg, seed, subject, fuzzBound(int64(subs), 0, fuzzMaxItems), fuzzIsAsync(amask, 0), false)
		requireWaitGroupDone(t, "feed after a closed Subscriber", &wg)

		if closed.terminals() != 0 || atomic.LoadInt32(&closed.nexts) != 0 {
			t.Errorf("closed subscriber received notifications")
		}

		live.check(t, "live subscriber")
	})
}

// FuzzSubjectUnicastReplayUnsubscribe unsubscribes the subscriber from inside the buffered replay.
func FuzzSubjectUnicastReplayUnsubscribe(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := ro.NewUnicastSubject[int](ro.UnicastSubjectUnlimitedBufferSize)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		stopAt := fuzzBound(int64(extra), 0, nItems-1)

		for i := 0; i < nItems; i++ {
			subject.Next(i)
		}

		rec := &subjectRecorder{}

		var sub ro.Subscriber[int]

		rec.onNext = func(value int) {
			if value == stopAt {
				sub.Unsubscribe()
				atomic.StoreInt32(&rec.unsubbed, 1)
			}
		}
		sub = ro.NewSubscriber(rec.observer())

		requireReturnsWithin(t, "Subscribe unsubscribing during replay", func() {
			subject.Subscribe(sub)
		})

		var wg sync.WaitGroup

		// Live items after the replay must not reach the unsubscribed observer.
		feedSubject(&wg, seed, subject, fuzzBound(int64(subs), 0, fuzzMaxItems), fuzzIsAsync(amask, 0), mask&1 == 1)
		requireWaitGroupDone(t, "feed after replay unsubscription", &wg)

		if got := int(atomic.LoadInt32(&rec.nexts)); got != stopAt+1 {
			t.Errorf("observer received %d values, want %d (nothing after the unsubscription)", got, stopAt+1)
		}

		rec.check(t, "subscriber")
	})
}

// FuzzSubjectUnicastResubscribe re-subscribes to a UnicastSubject after the first subscriber left.
func FuzzSubjectUnicastResubscribe(f *testing.F) {
	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		bufSize := int(extra%5) - 1
		subject := ro.NewUnicastSubject[int](bufSize)
		nGap := fuzzBound(int64(items), 0, fuzzMaxItems)
		nLive := fuzzBound(int64(subs), 0, fuzzMaxItems)

		first := &subjectRecorder{}
		firstSub := subject.Subscribe(first.observer())

		subject.Next(-1)
		firstSub.Unsubscribe()

		if subject.HasObserver() {
			t.Fatalf("observer still registered after Unsubscribe")
		}

		for i := 0; i < nGap; i++ {
			subject.Next(gapValueBase + i)
			fuzzJitter(seed, i)
		}

		second := &subjectRecorder{}
		requireReturnsWithin(t, "re-subscribe", func() { subject.Subscribe(second.observer()) })

		var wg sync.WaitGroup

		feedSubject(&wg, seed, subject, nLive, fuzzIsAsync(amask, 0), mask&1 == 1)
		requireWaitGroupDone(t, "feed after re-subscribe", &wg)

		got := second.snapshot()

		// Layout of what the second subscriber sees: replayed gap values (a suffix of what was
		// buffered, in order) followed by every live value.
		replayed := len(got) - nLive

		if replayed < 0 || replayed > nGap || (bufSize != ro.UnicastSubjectUnlimitedBufferSize && replayed > bufSize) {
			t.Fatalf("second subscriber saw %d replayed values for %d buffered (buffer %d): %v", replayed, nGap, bufSize, got)
		}

		for i := 0; i < replayed; i++ {
			if want := gapValueBase + nGap - replayed + i; got[i] != want {
				t.Fatalf("replayed value %d = %d, want %d: %v", i, got[i], want, got)
			}
		}

		for i := 0; i < nLive; i++ {
			if got[replayed+i] != i {
				t.Fatalf("live value %d = %d, want %d: %v", i, got[replayed+i], i, got)
			}
		}

		if second.terminals() != 1 {
			t.Errorf("second subscriber: %d terminals, want 1", second.terminals())
		}

		first.check(t, "first subscriber")
		second.check(t, "second subscriber")

		if atomic.LoadInt32(&first.nexts) != 1 {
			t.Errorf("first subscriber received values after leaving")
		}
	})
}

// FuzzSubjectLongLock parks one observer inside Next and checks that Subscribe and IsClosed
// on the same subject still return, instead of waiting for the observer.
func FuzzSubjectLongLock(f *testing.F) {
	f.Skip("race: subjects-long-lock; remove when fixed")

	addSubjectSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := newSubject(kind, extra)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		trigger := nItems - 1

		entered := make(chan struct{})
		release := make(chan struct{})

		var once sync.Once

		blocker := ro.OnNext(func(value int) {
			if value == trigger {
				once.Do(func() { close(entered) })
				<-release
			}
		})

		requireReturnsWithin(t, "subscribe blocker", func() { subject.Subscribe(blocker) })

		var wg sync.WaitGroup

		// The source's last item parks the blocker (an AsyncSubject only notifies it on completion).
		feedSubject(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), false)

		select {
		case <-entered:
		case <-time.After(fuzzDeadline):
			close(release)
			t.Fatalf("blocking observer was never reached")
		}

		var probes sync.WaitGroup

		nProbes := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		for i := 0; i < nProbes; i++ {
			probes.Add(1)

			go func(i int) {
				defer probes.Done()

				fuzzJitter(seed, i)

				if (mask>>(uint(i)%8))&1 == 1 {
					_ = subject.IsClosed()
				} else {
					subject.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
				}
			}(i)
		}

		returned := waitGroupDoneWithin(subjectShortDeadline, &probes)

		close(release)

		requireWaitGroupDone(t, "source after the observer was released", &wg)
		// Probes unblock once the lock is released, so waiting here leaks no goroutine.
		requireWaitGroupDone(t, "probes after release", &probes)

		if !returned {
			t.Fatalf("Subscribe/IsClosed blocked for more than %s while an observer was parked in Next", subjectShortDeadline)
		}
	})
}
