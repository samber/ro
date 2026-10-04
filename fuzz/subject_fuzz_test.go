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
	"sync"
	"testing"
	"time"

	"github.com/samber/ro"
)

const (
	// parkedObserverProbeTimeout bounds waits that must NOT be blocked by a slow observer. It is short
	// on purpose: the assertion is that the call returns while another goroutine is still stuck inside
	// an observer.
	parkedObserverProbeTimeout = 300 * time.Millisecond

	// reentrantValue is the value pushed by a re-entrant Next. Sources only emit 0..maxItems-1, so it
	// never collides with them.
	reentrantValue = 1 << 20

	// gapValueBase offsets values emitted while no observer is attached, so they can be told apart
	// from the values the source emits afterwards.
	gapValueBase = 1 << 10

	// lateValue is pushed to a terminated subject: no subscriber may receive it.
	lateValue = -42

	// maxProducers bounds the number of goroutines calling Next concurrently.
	maxProducers = 4

	// lowestBufferSize and highestBufferSize bound the buffer size of replay and unicast subjects:
	// -1 is the unlimited buffer, then zero and small positive sizes.
	lowestBufferSize  = -1
	highestBufferSize = 3
)

// subjectOfKind builds one of the five subject types from a fuzz byte:
// publish, behavior, replay, async and unicast. Replay and unicast take the buffer size.
func subjectOfKind(subjectKind uint8, bufferSize int) ro.Subject[int] {
	switch bounded(subjectKind, 0, 4) {
	case 0:
		return ro.NewPublishSubject[int]()
	case 1:
		return ro.NewBehaviorSubject(-1)
	case 2:
		return ro.NewReplaySubject[int](bufferSize)
	case 3:
		return ro.NewAsyncSubject[int]()
	default:
		return ro.NewUnicastSubject[int](bufferSize)
	}
}

// subjectSource is a source of `items` integers that feeds a subject, ending with a completion or,
// when failAtEnd is set, with an error. It yields the processor at positions chosen by yieldPattern.
func subjectSource(items int, asyncSource, failAtEnd bool, yieldPattern uint8) *source {
	numbers := newSource(items, asyncSource).yieldingWith(int64(yieldPattern))
	if failAtEnd {
		numbers = numbers.failingAtEnd(errInjectedFailure)
	}

	return numbers
}

// subjectFeed is a source piped into a subject from its own goroutine, so that a subject blocked
// in Next does not block the target.
type subjectFeed struct{ finished chan struct{} }

// feedSubject pipes numbers into subject. The subject ends with the source.
func feedSubject(subject ro.Subject[int], numbers *source) *subjectFeed {
	feed := &subjectFeed{finished: make(chan struct{})}

	go func() {
		defer close(feed.finished)

		subscription := numbers.observable().Subscribe(subject)
		numbers.waitForProducers()
		subscription.Unsubscribe()
	}()

	return feed
}

// expectFinished fails the test when the source did not end the subject in time.
func (feed *subjectFeed) expectFinished(t *testing.T, what string) {
	t.Helper()

	waitUntil(t, what+" to finish feeding the subject (deadlock)", func() bool { return isClosed(feed.finished) })
}

// subscribeProbes subscribes `count` fresh probes to subject one after the other, failing the test
// when a Subscribe does not return.
func subscribeProbes(t *testing.T, subject ro.Subject[int], count int) []*subscriberProbe {
	t.Helper()

	probes := make([]*subscriberProbe, count)
	for i := range probes {
		probes[i] = &subscriberProbe{}

		expectReturns(t, "Subscribe", func() { probes[i].subscribe(subject) })
	}

	return probes
}

// expectNoObserverLeft checks that a terminated subject released every observer.
func expectNoObserverLeft(t *testing.T, subject ro.Subject[int]) {
	t.Helper()

	if got := subject.CountObservers(); got != 0 {
		t.Fatalf("%d observers still registered after completion", got)
	}
}

// FuzzSubjectConcurrentSubscribeNext subscribes from many goroutines while a source emits and ends,
// on every kind of subject. The first `leavingSubscribers` subscribers unsubscribe right after subscribing.
//
// Invariant: every subscriber sees non-overlapping notifications and at most one terminal; a
// subscriber that stays sees exactly one terminal. Nothing deadlocks.
//
// Seeds: all arguments spread over their range; asyncSource and failAtEnd alternate.
func FuzzSubjectConcurrentSubscribeNext(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// subjectKind, bufferSize, subscribers, items, leavingSubscribers, yieldPattern, asyncSource, failAtEnd
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, subscribers, items, leavingSubscribers, yieldPattern uint8, asyncSource, failAtEnd bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))
		subscriberCount := bounded(subscribers, 1, maxWorkers)
		leavers := bounded(leavingSubscribers, 0, subscriberCount)

		probes := make([]*subscriberProbe, subscriberCount)

		var subscribing sync.WaitGroup

		for i := range probes {
			probes[i] = &subscriberProbe{}

			subscribing.Add(1)

			go func(i int) {
				defer subscribing.Done()

				yieldAt(int64(yieldPattern), i)
				probes[i].subscribe(subject)
				yieldAt(int64(yieldPattern), i+maxWorkers)

				if i < leavers {
					probes[i].unsubscribe()
				}
			}(i)
		}

		feed := feedSubject(subject, subjectSource(bounded(items, 0, maxItems), asyncSource, failAtEnd, yieldPattern))

		expectWaitGroupDone(t, "concurrent subscribe and next", &subscribing)
		feed.expectFinished(t, "source")

		for i, probe := range probes {
			probe.expectContract(t, "subscriber")

			// A subscriber that never left outlives the terminal, so it must have seen exactly one.
			if i >= leavers && probe.terminalCount() != 1 {
				t.Fatalf("subscriber %d: %d terminal notifications, want exactly 1", i, probe.terminalCount())
			}
		}
	})
}

// FuzzSubjectSelfUnsubscribeDuringBroadcast makes every subscriber unsubscribe from inside its own
// Next callback, once it received more than `unsubscribeAfter` values, while a source broadcasts.
//
// Invariant: no subscriber is notified after it unsubscribed itself, contracts hold, and no observer
// stays registered once the subject ended.
//
// Seeds: all arguments spread over their range; asyncSource alternates.
func FuzzSubjectSelfUnsubscribeDuringBroadcast(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// subjectKind, bufferSize, subscribers, items, unsubscribeAfter, yieldPattern, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, subscribers, items, unsubscribeAfter, yieldPattern uint8, asyncSource bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))
		count := bounded(items, 1, maxItems)
		threshold := bounded(unsubscribeAfter, 0, count-1)

		probes := make([]*subscriberProbe, bounded(subscribers, 1, maxWorkers))
		for i := range probes {
			probe := &subscriberProbe{}
			probe.onValue = func(int) {
				if probe.valueCount() > threshold {
					probe.unsubscribe()
				}
			}

			probes[i] = probe

			expectReturns(t, "Subscribe", func() { probe.subscribe(subject) })
		}

		feed := feedSubject(subject, subjectSource(count, asyncSource, false, yieldPattern))
		feed.expectFinished(t, "source")

		for _, probe := range probes {
			probe.expectContract(t, "subscriber")
			probe.expectSilentAfterUnsubscribe(t, "subscriber")
		}

		expectNoObserverLeft(t, subject)
	})
}

// FuzzSubjectOutsideUnsubscribeDuringBroadcast unsubscribes the first `leavingSubscribers` subscribers
// from another goroutine, while a source broadcasts.
//
// Invariant: contracts hold for every subscriber, and no observer stays registered once the subject
// ended.
//
// Seeds: all arguments spread over their range; asyncSource alternates.
func FuzzSubjectOutsideUnsubscribeDuringBroadcast(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// subjectKind, bufferSize, subscribers, items, leavingSubscribers, yieldPattern, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, subscribers, items, leavingSubscribers, yieldPattern uint8, asyncSource bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))
		probes := subscribeProbes(t, subject, bounded(subscribers, 1, maxWorkers))
		leavers := bounded(leavingSubscribers, 0, len(probes))

		var leaving sync.WaitGroup

		leaving.Add(1)

		go func() {
			defer leaving.Done()

			for i := 0; i < leavers; i++ {
				yieldAt(int64(yieldPattern), i+3*maxWorkers)
				probes[i].unsubscribe()
			}
		}()

		feed := feedSubject(subject, subjectSource(bounded(items, 1, maxItems), asyncSource, false, yieldPattern))

		expectWaitGroupDone(t, "unsubscribe during broadcast", &leaving)
		feed.expectFinished(t, "source")

		for _, probe := range probes {
			probe.expectContract(t, "subscriber")
		}

		expectNoObserverLeft(t, subject)
	})
}

// FuzzSubjectNextRacesTerminal races Next, called from `producers` goroutines, against two concurrent
// Complete or Error calls and the source's own terminal.
//
// Invariant: every subscriber sees exactly one terminal and no overlapping notification; the subject
// is closed afterwards and drops later values.
//
// Seeds: all arguments spread over their range; the booleans alternate at different rates.
func FuzzSubjectNextRacesTerminal(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// subjectKind, bufferSize, subscribers, items, producers, yieldPattern,
		// asyncSource, failAtEnd, firstTerminatorFails, secondTerminatorFails
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0, i%3 == 0, i%5 < 2, i%7 < 3}
	})

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, subscribers, items, producers, yieldPattern uint8, asyncSource, failAtEnd, firstTerminatorFails, secondTerminatorFails bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))
		count := bounded(items, 0, maxItems)
		producerCount := bounded(producers, 1, maxProducers)
		probes := subscribeProbes(t, subject, bounded(subscribers, 1, maxWorkers))

		var racing sync.WaitGroup

		// Producer 0 is the source, which also ends the subject; the other producers call Next directly.
		feed := feedSubject(subject, subjectSource(count, asyncSource, failAtEnd, yieldPattern))

		for producer := 1; producer < producerCount; producer++ {
			racing.Add(1)

			go func(producer int) {
				defer racing.Done()

				for i := 0; i < count; i++ {
					subject.Next(producer*gapValueBase + i)
					yieldAt(int64(yieldPattern), i+producer*7)
				}
			}(producer)
		}

		for terminator, fails := range []bool{firstTerminatorFails, secondTerminatorFails} {
			racing.Add(1)

			go func(terminator int, fails bool) {
				defer racing.Done()

				yieldAt(int64(yieldPattern), 5*maxWorkers+terminator)

				if fails {
					subject.Error(errInjectedFailure)
				} else {
					subject.Complete()
				}
			}(terminator, fails)
		}

		expectWaitGroupDone(t, "next racing terminal", &racing)
		feed.expectFinished(t, "source")

		expectTerminatedSubjectIsSilent(t, subject, probes)
	})
}

// expectTerminatedSubjectIsSilent checks that every probe saw exactly one terminal, that the subject
// is closed, and that a value pushed afterwards reaches nobody.
func expectTerminatedSubjectIsSilent(t *testing.T, subject ro.Subject[int], probes []*subscriberProbe) {
	t.Helper()

	valuesBefore := make([]int, len(probes))

	for i, probe := range probes {
		probe.expectContract(t, "subscriber")

		if probe.terminalCount() != 1 {
			t.Fatalf("subscriber %d: %d terminal notifications, want exactly 1", i, probe.terminalCount())
		}

		valuesBefore[i] = probe.valueCount()
	}

	if !subject.IsClosed() {
		t.Fatalf("subject is not closed after a terminal")
	}

	subject.Next(lateValue)

	for i, probe := range probes {
		if probe.valueCount() != valuesBefore[i] {
			t.Fatalf("subscriber %d received a Next after the subject terminated", i)
		}
	}
}

// reenterFromFirstNotification sends the first half of the items to subject before any subscriber exists, so
// that replaying subjects run the observer inside Subscribe. It then subscribes a probe whose first
// notification calls reenter, and feeds the other half.
func reenterFromFirstNotification(t *testing.T, subject ro.Subject[int], items, yieldPattern uint8, asyncSource, failAtEnd bool, reenter func()) {
	t.Helper()

	probe := &subscriberProbe{onFirstNotification: reenter}
	count := bounded(items, 0, maxItems)
	early := count / 2

	expectReturns(t, "re-entrant call", func() {
		for i := 0; i < early; i++ {
			subject.Next(i)
		}

		probe.subscribe(subject)
	})

	feed := feedSubject(subject, subjectSource(count-early, asyncSource, failAtEnd, yieldPattern))
	feed.expectFinished(t, "re-entrant call")

	probe.expectContract(t, "subscriber")
}

// reentrantSeeds generates the seeds of the three re-entrant targets.
func reentrantSeeds(i int) []any {
	// subjectKind, bufferSize, items, yieldPattern, asyncSource, failAtEnd
	return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0}
}

// FuzzSubjectReentrantIsClosed calls subject.IsClosed() from inside the first notification a subscriber
// receives, on every kind of subject.
//
// Invariant: the call returns (no deadlock on the subject lock) and the subscriber contract holds.
//
// Seeds: all arguments spread over their range; asyncSource and failAtEnd alternate.
func FuzzSubjectReentrantIsClosed(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSeeds(f, reentrantSeeds)

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, items, yieldPattern uint8, asyncSource, failAtEnd bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))

		reenterFromFirstNotification(t, subject, items, yieldPattern, asyncSource, failAtEnd, func() { _ = subject.IsClosed() })
	})
}

// FuzzSubjectReentrantNext calls subject.Next() from inside the first notification a subscriber
// receives, on every kind of subject.
//
// Invariant: the call returns (no deadlock on the subject lock) and the subscriber contract holds.
//
// Seeds: same as FuzzSubjectReentrantIsClosed.
func FuzzSubjectReentrantNext(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSeeds(f, reentrantSeeds)

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, items, yieldPattern uint8, asyncSource, failAtEnd bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))

		reenterFromFirstNotification(t, subject, items, yieldPattern, asyncSource, failAtEnd, func() { subject.Next(reentrantValue) })
	})
}

// FuzzSubjectReentrantSubscribe calls subject.Subscribe() from inside the first notification a
// subscriber receives, on every kind of subject.
//
// Invariant: the call returns (no deadlock on the subject lock) and the subscriber contract holds.
//
// Seeds: same as FuzzSubjectReentrantIsClosed.
func FuzzSubjectReentrantSubscribe(f *testing.F) {
	f.Skip("race: subjects-reentrant-lock; remove when fixed")

	fuzzSeeds(f, reentrantSeeds)

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, items, yieldPattern uint8, asyncSource, failAtEnd bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))

		reenterFromFirstNotification(t, subject, items, yieldPattern, asyncSource, failAtEnd, func() {
			subject.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
		})
	})
}

// FuzzSubjectUnicastClosedSubscriber subscribes an already unsubscribed Subscriber to a UnicastSubject
// that buffered `bufferedItems` values and may have terminated.
//
// Invariant: Subscribe returns, the closed subscriber receives nothing, and the subject stays usable
// by a live subscriber fed by a source.
//
// Seeds: all arguments spread over their range; asyncSource, terminatedBefore and failedBefore alternate.
func FuzzSubjectUnicastClosedSubscriber(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// bufferSize, bufferedItems, liveItems, yieldPattern, asyncSource, terminatedBefore, failedBefore
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, bufferSize, bufferedItems, liveItems, yieldPattern uint8, asyncSource, terminatedBefore, failedBefore bool) {
		subject := ro.NewUnicastSubject[int](bounded(bufferSize, lowestBufferSize, highestBufferSize))

		for i := 0; i < bounded(bufferedItems, 0, maxItems); i++ {
			subject.Next(i)
		}

		switch {
		case terminatedBefore && failedBefore:
			subject.Error(errInjectedFailure)
		case terminatedBefore:
			subject.Complete()
		}

		closed := &subscriberProbe{}
		closedSubscriber := ro.NewSubscriber(closed.observer())
		closedSubscriber.Unsubscribe()

		expectReturns(t, "Subscribe with a closed Subscriber", func() { subject.Subscribe(closedSubscriber) })

		live := &subscriberProbe{}

		expectReturns(t, "Subscribe after a closed Subscriber", func() { live.subscribe(subject) })

		feed := feedSubject(subject, subjectSource(bounded(liveItems, 0, maxItems), asyncSource, false, yieldPattern))
		feed.expectFinished(t, "source")

		if closed.terminalCount() != 0 || closed.valueCount() != 0 {
			t.Fatalf("closed subscriber received notifications")
		}

		live.expectContract(t, "live subscriber")
	})
}

// FuzzSubjectUnicastReplayUnsubscribe unsubscribes the only subscriber of a UnicastSubject from inside the
// replay of its `bufferedItems` buffered values, then feeds live items.
//
// Invariant: Subscribe returns, the subscriber receives exactly the values up to the one that
// unsubscribed it, and contracts hold.
//
// Seeds: all arguments spread over their range; asyncSource and failAtEnd alternate.
func FuzzSubjectUnicastReplayUnsubscribe(f *testing.F) {
	f.Skip("race: unicast-closed-subscriber-selfdeadlock; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// bufferedItems, unsubscribeAt, liveItems, yieldPattern, asyncSource, failAtEnd
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, bufferedItems, unsubscribeAt, liveItems, yieldPattern uint8, asyncSource, failAtEnd bool) {
		subject := ro.NewUnicastSubject[int](ro.UnicastSubjectUnlimitedBufferSize)
		buffered := bounded(bufferedItems, 1, maxItems)
		stopAt := bounded(unsubscribeAt, 0, buffered-1)

		for i := 0; i < buffered; i++ {
			subject.Next(i)
		}

		probe := &subscriberProbe{}

		var subscriber ro.Subscriber[int]

		probe.onValue = func(value int) {
			if value == stopAt {
				subscriber.Unsubscribe()
				probe.markUnsubscribed()
			}
		}
		subscriber = ro.NewSubscriber(probe.observer())

		expectReturns(t, "Subscribe unsubscribing during replay", func() { subject.Subscribe(subscriber) })

		// Live items after the replay must not reach the unsubscribed observer.
		feed := feedSubject(subject, subjectSource(bounded(liveItems, 0, maxItems), asyncSource, failAtEnd, yieldPattern))
		feed.expectFinished(t, "source")

		if got := probe.valueCount(); got != stopAt+1 {
			t.Fatalf("observer received %d values, want %d (nothing after the unsubscription)", got, stopAt+1)
		}

		probe.expectContract(t, "subscriber")
	})
}

// FuzzSubjectUnicastResubscribe subscribes to a UnicastSubject, unsubscribes, lets `gapItems` values pile up
// in its buffer, then subscribes a second subscriber and feeds `liveItems` live values.
//
// Invariant: the second subscriber sees a suffix of the buffered values (bounded by the buffer size),
// in order, then every live value, then one terminal; the first subscriber sees nothing after leaving.
//
// Seeds: all arguments spread over their range; asyncSource and failAtEnd alternate.
func FuzzSubjectUnicastResubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// bufferSize, gapItems, liveItems, yieldPattern, asyncSource, failAtEnd
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, bufferSize, gapItems, liveItems, yieldPattern uint8, asyncSource, failAtEnd bool) {
		size := bounded(bufferSize, lowestBufferSize, highestBufferSize)
		subject := ro.NewUnicastSubject[int](size)
		gap := bounded(gapItems, 0, maxItems)
		live := bounded(liveItems, 0, maxItems)

		first := &subscriberProbe{}
		firstSubscription := first.subscribe(subject)

		subject.Next(-1)
		firstSubscription.Unsubscribe()

		if subject.HasObserver() {
			t.Fatalf("observer still registered after Unsubscribe")
		}

		for i := 0; i < gap; i++ {
			subject.Next(gapValueBase + i)
			yieldAt(int64(yieldPattern), i)
		}

		second := &subscriberProbe{}

		expectReturns(t, "re-subscribe", func() { second.subscribe(subject) })

		feed := feedSubject(subject, subjectSource(live, asyncSource, failAtEnd, yieldPattern))
		feed.expectFinished(t, "source")

		expectReplayedSuffixThenLiveValues(t, second.received(), gap, live, size)

		if second.terminalCount() != 1 {
			t.Fatalf("second subscriber: %d terminals, want 1", second.terminalCount())
		}

		first.expectContract(t, "first subscriber")
		second.expectContract(t, "second subscriber")

		if first.valueCount() != 1 {
			t.Fatalf("first subscriber received values after leaving")
		}
	})
}

// expectReplayedSuffixThenLiveValues checks what a late subscriber of a buffering subject saw: a suffix of
// the `gap` buffered values, in order and no longer than the buffer, then every live value 0..live-1.
func expectReplayedSuffixThenLiveValues(t *testing.T, got []int, gap, live, bufferSize int) {
	t.Helper()

	replayed := len(got) - live

	if replayed < 0 || replayed > gap || (bufferSize != ro.UnicastSubjectUnlimitedBufferSize && replayed > bufferSize) {
		t.Fatalf("second subscriber saw %d replayed values for %d buffered (buffer %d): %v", replayed, gap, bufferSize, got)
	}

	for i := 0; i < replayed; i++ {
		if want := gapValueBase + gap - replayed + i; got[i] != want {
			t.Fatalf("replayed value %d = %d, want %d: %v", i, got[i], want, got)
		}
	}

	for i := 0; i < live; i++ {
		if got[replayed+i] != i {
			t.Fatalf("live value %d = %d, want %d: %v", i, got[replayed+i], i, got)
		}
	}
}

// FuzzSubjectLongLock parks one subscriber inside Next, on the source's last item, then calls IsClosed (the
// first `isClosedProbes` goroutines) and Subscribe followed by Unsubscribe (the others) on the same subject.
//
// Invariant: those calls return within parkedObserverProbeTimeout while the observer is parked; the
// subject lock is not held while an observer runs.
//
// Seeds: all arguments spread over their range; asyncSource alternates.
func FuzzSubjectLongLock(f *testing.F) {
	f.Skip("race: subjects-long-lock; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// subjectKind, bufferSize, items, probes, isClosedProbes, yieldPattern, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, subjectKind, bufferSize, items, probes, isClosedProbes, yieldPattern uint8, asyncSource bool) {
		subject := subjectOfKind(subjectKind, bounded(bufferSize, lowestBufferSize, highestBufferSize))
		count := bounded(items, 1, maxItems)
		parked, release := parkObserverOnValue(t, subject, count-1)

		// An AsyncSubject only notifies the parked observer on completion, which the source's end triggers.
		feed := feedSubject(subject, subjectSource(count, asyncSource, false, yieldPattern))

		waitForParkedObserver(t, parked, release)

		probeCount := bounded(probes, 1, maxWorkers)
		returned, probing := probeSubjectWhileParked(subject, probeCount, bounded(isClosedProbes, 0, probeCount), yieldPattern)

		close(release)

		feed.expectFinished(t, "source after the observer was released")
		// Probes unblock once the lock is released, so waiting here leaks no goroutine.
		expectWaitGroupDone(t, "probes after release", probing)

		if !returned {
			t.Fatalf("Subscribe/IsClosed blocked for more than %s while an observer was parked in Next", parkedObserverProbeTimeout)
		}
	})
}

// parkObserverOnValue subscribes an observer that blocks inside Next when it receives value. The first
// channel closes once the observer is parked; closing the second one releases it.
func parkObserverOnValue(t *testing.T, subject ro.Subject[int], value int) (parked <-chan struct{}, release chan struct{}) {
	t.Helper()

	enteredNext := make(chan struct{})
	release = make(chan struct{})

	var once sync.Once

	blocker := ro.OnNext(func(received int) {
		if received == value {
			once.Do(func() { close(enteredNext) })
			<-release
		}
	})

	expectReturns(t, "subscribe the observer to park", func() { subject.Subscribe(blocker) })

	return enteredNext, release
}

// waitForParkedObserver waits until the observer is parked, releasing it first when it never is.
func waitForParkedObserver(t *testing.T, parked <-chan struct{}, release chan struct{}) {
	t.Helper()

	select {
	case <-parked:
	case <-time.After(waitDeadline):
		close(release)
		t.Fatalf("blocking observer was never reached")
	}
}

// probeSubjectWhileParked starts `count` goroutines calling the subject, the first `isClosedCount` with IsClosed
// and the others with Subscribe then Unsubscribe. It reports whether all returned within
// parkedObserverProbeTimeout, and returns the group to wait on once the observer is released.
func probeSubjectWhileParked(subject ro.Subject[int], count, isClosedCount int, yieldPattern uint8) (returned bool, probing *sync.WaitGroup) {
	probing = &sync.WaitGroup{}

	for i := 0; i < count; i++ {
		probing.Add(1)

		go func(i int) {
			defer probing.Done()

			yieldAt(int64(yieldPattern), i)

			if i < isClosedCount {
				_ = subject.IsClosed()
			} else {
				subject.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
			}
		}(i)
	}

	return waitGroupFinished(probing, parkedObserverProbeTimeout), probing
}
