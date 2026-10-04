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
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro/internal/xtest"
)

const (
	// fuzzSubjKindCount is the number of subject implementations a fuzz input can select.
	fuzzSubjKindCount = 5

	// fuzzSubjShortDeadline bounds waits that must NOT be blocked by a slow observer. It is
	// short on purpose: the assertion is that the call returns while another goroutine is still
	// stuck inside an observer.
	fuzzSubjShortDeadline = 300 * time.Millisecond

	// fuzzSubjReentrantValue is the value pushed by a re-entrant Next. Sources only emit
	// 0..n-1 with n <= fuzzMaxItems, so it never collides with them.
	fuzzSubjReentrantValue = 1 << 20

	// fuzzSubjGapBase offsets values emitted while no observer is attached, so they can be
	// told apart from values emitted by the source afterwards.
	fuzzSubjGapBase = 1 << 10

	// fuzzSubjMaxProducers bounds the number of concurrent Next callers.
	fuzzSubjMaxProducers = 4
)

var errFuzzSubj = errors.New("fuzzSubj: boom")

// fuzzSubjNew builds one of the 5 subject types. bufSize feeds the replay and unicast buffers
// and covers -1 (unlimited), 0 and small positive values.
func fuzzSubjNew(kind, bufSize uint8) Subject[int] {
	size := int(bufSize%5) - 1

	switch kind % fuzzSubjKindCount {
	case 0:
		return NewPublishSubject[int]()
	case 1:
		return NewBehaviorSubject(-1)
	case 2:
		return NewReplaySubject[int](size)
	case 3:
		return NewAsyncSubject[int]()
	default:
		return NewUnicastSubject[int](size)
	}
}

// fuzzSubjFeed pipes fuzzSource(0..n-1) into subject and terminates it with Complete, or Error when
// fail is set. A synchronous source emits inside Subscribe, an asynchronous one from its own goroutine.
// wg is released once the subject has been terminated.
func fuzzSubjFeed(wg *sync.WaitGroup, seed int64, subject Subject[int], n int, async, fail bool) {
	done := make(chan struct{})

	wg.Add(1)

	go func() {
		defer wg.Done()

		sub := fuzzSource(seed, n, async).Subscribe(NewObserver(
			subject.Next,
			subject.Error,
			func() {
				if fail {
					subject.Error(errFuzzSubj)
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

// fuzzSubjRecorder records what one subscriber observes and checks the observer contract.
type fuzzSubjRecorder struct {
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

func (r *fuzzSubjRecorder) terminals() int {
	return int(atomic.LoadInt32(&r.errs) + atomic.LoadInt32(&r.completes))
}

func (r *fuzzSubjRecorder) begin() {
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

func (r *fuzzSubjRecorder) observer() Observer[int] {
	return NewObserver(
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

func (r *fuzzSubjRecorder) snapshot() []int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]int(nil), r.values...)
}

// check asserts the invariants that hold for every subscriber, whatever the interleaving.
func (r *fuzzSubjRecorder) check(t *testing.T, name string) {
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

// fuzzSubjWait reports whether wg finished within d, so that a deadlock fails the test instead of hanging it.
func fuzzSubjWait(d time.Duration, wg *sync.WaitGroup) bool {
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

// fuzzSubjWaitOrFail fails the test when wg does not finish within fuzzDeadline.
func fuzzSubjWaitOrFail(t *testing.T, what string, wg *sync.WaitGroup) {
	t.Helper()

	if !fuzzSubjWait(fuzzDeadline, wg) {
		t.Fatalf("deadlock: %s did not finish within %s", what, fuzzDeadline)
	}
}

// fuzzSubjWithin runs fn in its own goroutine and fails the test when it does not return in time.
func fuzzSubjWithin(t *testing.T, what string, fn func()) {
	t.Helper()

	var wg sync.WaitGroup

	wg.Add(1)

	go func() {
		defer wg.Done()

		fn()
	}()

	fuzzSubjWaitOrFail(t, what, &wg)
}

// fuzzSubjHolder hands a Subscription to a callback that may run before Subscribe returns.
type fuzzSubjHolder struct {
	mu  sync.Mutex
	sub Subscription
}

func (h *fuzzSubjHolder) set(sub Subscription) {
	h.mu.Lock()
	h.sub = sub
	h.mu.Unlock()
}

func (h *fuzzSubjHolder) get() Subscription {
	h.mu.Lock()
	defer h.mu.Unlock()

	return h.sub
}

func fuzzSubjSeeds(f *testing.F) {
	f.Helper()

	xtest.AddSeeds(f, func(i int) []any {
		return []any{int64(i), uint8(i), uint8(i / 5), uint8(i / 3), uint8(i * 37), uint8(i * 11), uint8(i * 13)}
	})
}

// FuzzSubjConcurrentSubscribeNext subscribes from many goroutines while a source emits and terminates.
func FuzzSubjConcurrentSubscribeNext(f *testing.F) {
	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := fuzzSubjNew(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)

		recorders := make([]*fuzzSubjRecorder, nSubs)
		unsubscribed := make([]bool, nSubs)

		var wg sync.WaitGroup

		for i := range recorders {
			recorders[i] = &fuzzSubjRecorder{}
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

		fuzzSubjFeed(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), mask&0x80 != 0)

		fuzzSubjWaitOrFail(t, "concurrent subscribe + next", &wg)

		for i, r := range recorders {
			r.check(t, "subscriber")

			// A subscriber that never unsubscribed outlives the terminal, so it must have seen exactly one.
			if !unsubscribed[i] && r.terminals() != 1 {
				t.Errorf("subscriber %d: %d terminal notifications, want exactly 1", i, r.terminals())
			}
		}
	})
}

// FuzzSubjUnsubscribeDuringBroadcast unsubscribes from another goroutine and from inside onNext.
func FuzzSubjUnsubscribeDuringBroadcast(f *testing.F) {
	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := fuzzSubjNew(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		unsubAfter := fuzzBound(int64(extra), 0, nItems-1)

		recorders := make([]*fuzzSubjRecorder, nSubs)
		holders := make([]*fuzzSubjHolder, nSubs)

		for i := range recorders {
			r := &fuzzSubjRecorder{}
			h := &fuzzSubjHolder{}

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

			fuzzSubjWithin(t, "subscribe", func() {
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

		fuzzSubjFeed(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), false)

		fuzzSubjWaitOrFail(t, "unsubscribe during broadcast", &wg)

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

// FuzzSubjNextRacesTerminal races Next against Complete and Error from several goroutines.
func FuzzSubjNextRacesTerminal(f *testing.F) {
	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := fuzzSubjNew(kind, extra)
		nSubs := fuzzBound(int64(subs), 1, fuzzMaxGoroutines)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)
		nProducers := fuzzBound(int64(extra), 1, fuzzSubjMaxProducers)

		recorders := make([]*fuzzSubjRecorder, nSubs)
		for i := range recorders {
			r := &fuzzSubjRecorder{}
			recorders[i] = r

			fuzzSubjWithin(t, "subscribe", func() {
				subject.Subscribe(r.observer())
			})
		}

		var wg sync.WaitGroup

		// Producer 0 is a source that also terminates the subject; the others call Next directly.
		// Together with the two terminators below, up to 3 terminal calls race.
		fuzzSubjFeed(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), mask&0x40 != 0)

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
					subject.Error(errFuzzSubj)
				} else {
					subject.Complete()
				}
			}(k)
		}

		fuzzSubjWaitOrFail(t, "next vs terminal", &wg)

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

// fuzzSubjReentrant calls op from inside the first notification an observer receives.
// op: 0 = IsClosed, 1 = Next, 2 = Subscribe.
func fuzzSubjReentrant(f *testing.F, op int) {
	f.Helper()
	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := fuzzSubjNew(kind, extra)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)
		pre := nItems / 2

		rec := &fuzzSubjRecorder{}
		rec.onEvent = func() {
			switch op {
			case 0:
				_ = subject.IsClosed()
			case 1:
				subject.Next(fuzzSubjReentrantValue)
			default:
				subject.Subscribe(OnNext(func(int) {})).Unsubscribe()
			}
		}

		var wg sync.WaitGroup

		fuzzSubjWithin(t, "re-entrant call", func() {
			// Items sent before Subscribe exercise the replay paths, which run the observer under Subscribe.
			for i := 0; i < pre; i++ {
				subject.Next(i)
			}

			subject.Subscribe(rec.observer())
		})

		fuzzSubjFeed(&wg, seed, subject, nItems-pre, fuzzIsAsync(amask, 0), mask&1 == 1)

		fuzzSubjWaitOrFail(t, "re-entrant call", &wg)

		rec.check(t, "subscriber")
	})
}

// FuzzSubjReentrantIsClosed calls subject.IsClosed() from inside onNext.
func FuzzSubjReentrantIsClosed(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSubjReentrant(f, 0)
}

// FuzzSubjReentrantNext calls subject.Next() from inside onNext.
func FuzzSubjReentrantNext(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSubjReentrant(f, 1)
}

// FuzzSubjReentrantSubscribe calls subject.Subscribe() from inside onNext.
func FuzzSubjReentrantSubscribe(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSubjReentrant(f, 2)
}

// FuzzSubjUnicastClosedSubscriber subscribes an already-closed Subscriber to a UnicastSubject.
func FuzzSubjUnicastClosedSubscriber(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := NewUnicastSubject[int](int(extra%5) - 1)
		nItems := fuzzBound(int64(items), 0, fuzzMaxItems)

		for i := 0; i < nItems; i++ {
			subject.Next(i)
		}

		switch mask % 3 {
		case 1:
			subject.Complete()
		case 2:
			subject.Error(errFuzzSubj)
		}

		closed := &fuzzSubjRecorder{}
		sub := NewSubscriber(closed.observer())
		sub.Unsubscribe()

		fuzzSubjWithin(t, "Subscribe with a closed Subscriber", func() {
			subject.Subscribe(sub)
		})

		// The subject must stay usable afterwards.
		live := &fuzzSubjRecorder{}

		fuzzSubjWithin(t, "Subscribe after a closed Subscriber", func() {
			subject.Subscribe(live.observer())
		})

		var wg sync.WaitGroup

		fuzzSubjFeed(&wg, seed, subject, fuzzBound(int64(subs), 0, fuzzMaxItems), fuzzIsAsync(amask, 0), false)
		fuzzSubjWaitOrFail(t, "feed after a closed Subscriber", &wg)

		if closed.terminals() != 0 || atomic.LoadInt32(&closed.nexts) != 0 {
			t.Errorf("closed subscriber received notifications")
		}

		live.check(t, "live subscriber")
	})
}

// FuzzSubjUnicastReplayUnsubscribe unsubscribes the subscriber from inside the buffered replay.
func FuzzSubjUnicastReplayUnsubscribe(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		stopAt := fuzzBound(int64(extra), 0, nItems-1)

		for i := 0; i < nItems; i++ {
			subject.Next(i)
		}

		rec := &fuzzSubjRecorder{}

		var sub Subscriber[int]

		rec.onNext = func(value int) {
			if value == stopAt {
				sub.Unsubscribe()
				atomic.StoreInt32(&rec.unsubbed, 1)
			}
		}
		sub = NewSubscriber(rec.observer())

		fuzzSubjWithin(t, "Subscribe unsubscribing during replay", func() {
			subject.Subscribe(sub)
		})

		var wg sync.WaitGroup

		// Live items after the replay must not reach the unsubscribed observer.
		fuzzSubjFeed(&wg, seed, subject, fuzzBound(int64(subs), 0, fuzzMaxItems), fuzzIsAsync(amask, 0), mask&1 == 1)
		fuzzSubjWaitOrFail(t, "feed after replay unsubscription", &wg)

		if got := int(atomic.LoadInt32(&rec.nexts)); got != stopAt+1 {
			t.Errorf("observer received %d values, want %d (nothing after the unsubscription)", got, stopAt+1)
		}

		rec.check(t, "subscriber")
	})
}

// FuzzSubjUnicastResubscribe re-subscribes to a UnicastSubject after the first subscriber left.
func FuzzSubjUnicastResubscribe(f *testing.F) {
	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		bufSize := int(extra%5) - 1
		subject := NewUnicastSubject[int](bufSize)
		nGap := fuzzBound(int64(items), 0, fuzzMaxItems)
		nLive := fuzzBound(int64(subs), 0, fuzzMaxItems)

		first := &fuzzSubjRecorder{}
		firstSub := subject.Subscribe(first.observer())

		subject.Next(-1)
		firstSub.Unsubscribe()

		if subject.HasObserver() {
			t.Fatalf("observer still registered after Unsubscribe")
		}

		for i := 0; i < nGap; i++ {
			subject.Next(fuzzSubjGapBase + i)
			fuzzJitter(seed, i)
		}

		second := &fuzzSubjRecorder{}
		fuzzSubjWithin(t, "re-subscribe", func() { subject.Subscribe(second.observer()) })

		var wg sync.WaitGroup

		fuzzSubjFeed(&wg, seed, subject, nLive, fuzzIsAsync(amask, 0), mask&1 == 1)
		fuzzSubjWaitOrFail(t, "feed after re-subscribe", &wg)

		got := second.snapshot()

		// Layout of what the second subscriber sees: replayed gap values (a suffix of what was
		// buffered, in order) followed by every live value.
		replayed := len(got) - nLive

		if replayed < 0 || replayed > nGap || (bufSize != UnicastSubjectUnlimitedBufferSize && replayed > bufSize) {
			t.Fatalf("second subscriber saw %d replayed values for %d buffered (buffer %d): %v", replayed, nGap, bufSize, got)
		}

		for i := 0; i < replayed; i++ {
			if want := fuzzSubjGapBase + nGap - replayed + i; got[i] != want {
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

// FuzzSubjLongLock parks one observer inside Next and checks that Subscribe and IsClosed
// on the same subject still return, instead of waiting for the observer.
func FuzzSubjLongLock(f *testing.F) {
	f.Skip("race: subjects-long-lock; remove when fixed")

	fuzzSubjSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, kind, subs, items, mask, extra, amask uint8) {
		t.Parallel()

		subject := fuzzSubjNew(kind, extra)
		nItems := fuzzBound(int64(items), 1, fuzzMaxItems)
		trigger := nItems - 1

		entered := make(chan struct{})
		release := make(chan struct{})

		var once sync.Once

		blocker := OnNext(func(value int) {
			if value == trigger {
				once.Do(func() { close(entered) })
				<-release
			}
		})

		fuzzSubjWithin(t, "subscribe blocker", func() { subject.Subscribe(blocker) })

		var wg sync.WaitGroup

		// The source's last item parks the blocker (an AsyncSubject only notifies it on completion).
		fuzzSubjFeed(&wg, seed, subject, nItems, fuzzIsAsync(amask, 0), false)

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
					subject.Subscribe(OnNext(func(int) {})).Unsubscribe()
				}
			}(i)
		}

		returned := fuzzSubjWait(fuzzSubjShortDeadline, &probes)

		close(release)

		fuzzSubjWaitOrFail(t, "source after the observer was released", &wg)
		// Probes unblock once the lock is released, so waiting here leaks no goroutine.
		fuzzSubjWaitOrFail(t, "probes after release", &probes)

		if !returned {
			t.Fatalf("Subscribe/IsClosed blocked for more than %s while an observer was parked in Next", fuzzSubjShortDeadline)
		}
	})
}
