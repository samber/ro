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
	"fmt"
	"sync"
	"testing"

	"github.com/samber/ro"
)

const (
	// secondSourceValueBase offsets the values of a second source so they never collide with the first.
	secondSourceValueBase = 1000

	// replayBufferLimit is the largest replay buffer a fuzz input can request.
	replayBufferLimit = 8

	// loopedSubscribers is the number of goroutines of FuzzConnectableConnectSubscribeLoop.
	loopedSubscribers = 16

	// maxLoopBursts is the number of Connect and Subscribe rounds each goroutine runs per iteration.
	maxLoopBursts = 4
)

// controlledSource is a source whose subscriptions are counted, driven either by the target
// (hand-fed: every subscription registers a destination the target pushes values to) or by itself
// (self-driven: a synchronous or asynchronous newSource that completes on its own).
type controlledSource struct {
	counter subscriptionCounter

	// selfDriven is the wrapped source, or nil for a hand-fed source.
	selfDriven *source

	mu           sync.Mutex
	destinations map[int]ro.Observer[int]
	nextID       int
}

// newControlledSource builds a self-driven source of `items` values when selfDriven is set, a hand-fed
// source otherwise. asyncSource only applies to the self-driven kind.
func newControlledSource(items int, selfDriven, asyncSource bool, yieldPattern uint8) *controlledSource {
	controlled := &controlledSource{}
	if selfDriven {
		controlled.selfDriven = newSource(items, asyncSource).yieldingWith(int64(yieldPattern))
	}

	return controlled
}

func (c *controlledSource) observable() ro.Observable[int] {
	if c.selfDriven != nil {
		return countSubscriptions(&c.counter, c.selfDriven.observable())
	}

	return countSubscriptions(&c.counter, ro.NewUnsafeObservable(func(destination ro.Observer[int]) ro.Teardown {
		c.mu.Lock()
		if c.destinations == nil {
			c.destinations = map[int]ro.Observer[int]{}
		}

		id := c.nextID
		c.nextID++
		c.destinations[id] = destination
		c.mu.Unlock()

		return func() {
			c.mu.Lock()
			delete(c.destinations, id)
			c.mu.Unlock()
		}
	}))
}

func (c *controlledSource) currentDestinations() []ro.Observer[int] {
	c.mu.Lock()
	defer c.mu.Unlock()

	destinations := make([]ro.Observer[int], 0, len(c.destinations))
	for _, destination := range c.destinations {
		destinations = append(destinations, destination)
	}

	return destinations
}

// emit pushes value to every current subscription of a hand-fed source.
func (c *controlledSource) emit(value int) {
	for _, destination := range c.currentDestinations() {
		destination.Next(value)
	}
}

// drive pushes 0..items-1 to a hand-fed source, then ends it when endSource is set, with an error when
// failAtEnd is set. A self-driven source emits and completes on its own, so drive does nothing.
func (c *controlledSource) drive(items int, yieldPattern uint8, endSource, failAtEnd bool) {
	if c.selfDriven != nil {
		return
	}

	for i := 0; i < items; i++ {
		yieldAt(int64(yieldPattern), i)
		c.emit(i)
	}

	if !endSource {
		return
	}

	for _, destination := range c.currentDestinations() {
		if failAtEnd {
			destination.Error(errInjectedFailure)
		} else {
			destination.Complete()
		}
	}
}

// FuzzConnectableConcurrent races Subscribe, Connect and Unsubscribe from several goroutines on one
// ConnectableObservable, while a hand-fed or self-driven source emits. The first `leavingSubscribers`
// subscribers unsubscribe right after subscribing, the first `leavingConnectors` connectors cancel
// their connection right after connecting.
//
// Invariant: every subscriber sees non-overlapping notifications and at most one terminal, and
// nothing deadlocks.
//
// Seeds: all arguments spread over their range; the booleans alternate at different rates.
func FuzzConnectableConcurrent(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, subscribers, leavingSubscribers, connectors, leavingConnectors, yieldPattern,
		// selfDrivenSource, asyncSource, endSource, failAtEnd, resetOnDisconnect
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), seedByte(i, 5), i%2 == 0, i%3 == 0, i%4 < 2, i%5 < 2, i%7 < 3}
	})

	f.Fuzz(func(t *testing.T, items, subscribers, leavingSubscribers, connectors, leavingConnectors, yieldPattern uint8, selfDrivenSource, asyncSource, endSource, failAtEnd, resetOnDisconnect bool) {
		count := bounded(items, 0, maxItems)
		subscriberCount := bounded(subscribers, 0, 2*maxWorkers)
		connectorCount := bounded(connectors, 0, 2*maxWorkers)

		numbers := newControlledSource(count, selfDrivenSource, asyncSource, yieldPattern)
		connectable := ro.ConnectableWithConfig(numbers.observable(), ro.ConnectableConfig[int]{
			Connector:         ro.NewPublishSubject[int],
			ResetOnDisconnect: resetOnDisconnect,
		})

		var wg sync.WaitGroup

		wg.Add(1)

		go func() {
			defer wg.Done()

			numbers.drive(count, yieldPattern, endSource, failAtEnd)
		}()

		probes := runConnectableSubscribers(&wg, connectable, subscriberCount, bounded(leavingSubscribers, 0, subscriberCount), yieldPattern)
		runConnectableConnectors(&wg, connectable, connectorCount, bounded(leavingConnectors, 0, connectorCount), yieldPattern)

		expectWaitGroupDone(t, "concurrent Connect, Subscribe and Unsubscribe", &wg)

		for _, probe := range probes {
			probe.expectContract(t, "subscriber")
		}
	})
}

// runConnectableSubscribers starts `count` goroutines that subscribe to connectable, the first
// `leaving` of them unsubscribing right away. It returns their probes.
func runConnectableSubscribers(wg *sync.WaitGroup, connectable ro.ConnectableObservable[int], count, leaving int, yieldPattern uint8) []*subscriberProbe {
	probes := make([]*subscriberProbe, count)

	for i := range probes {
		probes[i] = &subscriberProbe{}

		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			yieldAt(int64(yieldPattern), i)
			probes[i].subscribe(connectable)

			if i < leaving {
				yieldAt(int64(yieldPattern), i+maxWorkers)
				probes[i].unsubscribe()
			}
		}(i)
	}

	return probes
}

// runConnectableConnectors starts `count` goroutines that connect connectable, the first `leaving` of them
// cancelling their connection right away.
func runConnectableConnectors(wg *sync.WaitGroup, connectable ro.ConnectableObservable[int], count, leaving int, yieldPattern uint8) {
	for i := 0; i < count; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			yieldAt(int64(yieldPattern), i+2*maxWorkers)
			connection := connectable.Connect()

			if i < leaving {
				yieldAt(int64(yieldPattern), i+3*maxWorkers)
				connection.Unsubscribe()
			}
		}(i)
	}
}

// FuzzConnectableSyncReconnect connects concurrently, possibly after the source already completed, on a
// ConnectableObservable that resets on disconnect. The first `connectingFirstWorkers` workers Connect
// before they Subscribe; every worker Connects after subscribing.
//
// Invariant: every connection feeds a fresh subject. With a synchronous source, a subscriber that
// terminated saw the whole sequence, and one more connection delivers the whole sequence to a new
// subscriber. An asynchronous source may legitimately be joined mid-stream.
//
// Seeds: all arguments spread over their range; asyncSource alternates.
func FuzzConnectableSyncReconnect(f *testing.F) {
	f.Skip("race: connectable-stale-subject-after-sync-complete; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, workers, connectingFirstWorkers, yieldPattern, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, workers, connectingFirstWorkers, yieldPattern uint8, asyncSource bool) {
		count := bounded(items, 0, maxItems)
		workerCount := bounded(workers, 1, maxWorkers)
		connectingFirst := bounded(connectingFirstWorkers, 0, workerCount)

		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))
		connectable := ro.ConnectableWithConfig(numbers.observable(), ro.ConnectableConfig[int]{
			Connector:         ro.NewPublishSubject[int],
			ResetOnDisconnect: true,
		})

		probes := make([]*subscriberProbe, workerCount)

		var wg sync.WaitGroup

		for worker := range probes {
			probes[worker] = &subscriberProbe{}

			wg.Add(1)

			go func(worker int) {
				defer wg.Done()

				yieldAt(int64(yieldPattern), worker)

				if worker < connectingFirst {
					connectable.Connect()
				}

				probes[worker].subscribe(connectable)

				yieldAt(int64(yieldPattern), worker+maxWorkers)
				connectable.Connect()
			}(worker)
		}

		expectWaitGroupDone(t, "concurrent reconnect", &wg)

		for worker, probe := range probes {
			probe.expectContract(t, "subscriber")

			// With a synchronous source, a terminal without the whole sequence means the subscriber was
			// attached to a subject that had already completed, i.e. the connection did not get a fresh one.
			if !asyncSource && probe.terminalCount() > 0 && probe.valueCount() != count {
				t.Fatalf("worker %d: completed with %d values, want %d (connection fed a stale subject)", worker, probe.valueCount(), count)
			}
		}

		if !asyncSource {
			expectOneMoreConnectionDeliversEverything(t, connectable, count)
		}
	})
}

// expectOneMoreConnectionDeliversEverything connects a quiescent connectable once more and checks that a
// new subscriber receives the whole sequence and its terminal.
func expectOneMoreConnectionDeliversEverything(t *testing.T, connectable ro.ConnectableObservable[int], count int) {
	t.Helper()

	last := &subscriberProbe{}
	last.subscribe(connectable)
	connectable.Connect()

	if last.valueCount() != count || last.terminalCount() == 0 {
		t.Fatalf("final connection delivered %d/%d values, terminated=%v", last.valueCount(), count, last.terminalCount() > 0)
	}
}

// reentrantAction is what a re-entrant target does from inside a subscriber callback; inner is a probe
// it may subscribe.
type reentrantAction func(connectable ro.ConnectableObservable[int], inner *subscriberProbe)

// reenterConnectableFromCallback subscribes an outer probe to a connectable fed by a source of `items` values,
// and, when the outer probe receives the value `trigger`, calls reenter from inside that callback. The
// outer probe subscribes and connects from its own goroutine.
func reenterConnectableFromCallback(t *testing.T, items, trigger, yieldPattern uint8, asyncSource, resetOnDisconnect bool, reenter reentrantAction) {
	t.Helper()

	count := bounded(items, 1, maxItems)
	triggerValue := bounded(trigger, 0, count-1)

	numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))
	connectable := ro.ConnectableWithConfig(numbers.observable(), ro.ConnectableConfig[int]{
		Connector:         ro.NewPublishSubject[int],
		ResetOnDisconnect: resetOnDisconnect,
	})

	outer := &subscriberProbe{}
	inner := &subscriberProbe{}

	var fired sync.Once

	outer.onValue = func(value int) {
		if value == triggerValue {
			fired.Do(func() {
				yieldAt(int64(yieldPattern), value)
				reenter(connectable, inner)
			})
		}
	}

	var wg sync.WaitGroup

	wg.Add(1)

	go func() {
		defer wg.Done()

		outer.subscribe(connectable)
		connectable.Connect()
	}()

	scenario := fmt.Sprintf("async=%v resetOnDisconnect=%v", asyncSource, resetOnDisconnect)

	expectWaitGroupDone(t, "re-entrant call from a callback ("+scenario+")", &wg)
	waitUntil(t, "outer subscriber to terminate ("+scenario+")", func() bool { return outer.terminalCount() > 0 })

	outer.expectContract(t, "outer")
	inner.expectContract(t, "inner")
}

// FuzzConnectableConnectInsideCallback calls Connect from inside a subscriber callback, while the source is
// emitting, for a synchronous or asynchronous source.
//
// Invariant: Connect returns (no deadlock on the connectable lock), the outer subscriber terminates,
// and contracts hold.
//
// Seeds: all arguments spread over their range; asyncSource and resetOnDisconnect alternate.
func FuzzConnectableConnectInsideCallback(f *testing.F) {
	f.Skip("race: connectable-reentrant-deadlock (Connect holds mu across sync emission; async Subscribe-in-callback hangs the subject); remove when fixed")

	fuzzSeeds(f, reentrantConnectableSeeds)

	f.Fuzz(func(t *testing.T, items, trigger, yieldPattern uint8, asyncSource, resetOnDisconnect bool) {
		reenterConnectableFromCallback(t, items, trigger, yieldPattern, asyncSource, resetOnDisconnect,
			func(connectable ro.ConnectableObservable[int], _ *subscriberProbe) { connectable.Connect() })
	})
}

// FuzzConnectableSubscribeInsideCallback calls Subscribe from inside a subscriber callback, while the source is
// emitting, for a synchronous or asynchronous source.
//
// Invariant: Subscribe returns (no deadlock on the subject lock), the outer subscriber terminates,
// and contracts hold for the outer and the inner subscriber.
//
// Seeds: same as FuzzConnectableConnectInsideCallback.
func FuzzConnectableSubscribeInsideCallback(f *testing.F) {
	f.Skip("race: connectable-reentrant-deadlock (Connect holds mu across sync emission; async Subscribe-in-callback hangs the subject); remove when fixed")

	fuzzSeeds(f, reentrantConnectableSeeds)

	f.Fuzz(func(t *testing.T, items, trigger, yieldPattern uint8, asyncSource, resetOnDisconnect bool) {
		reenterConnectableFromCallback(t, items, trigger, yieldPattern, asyncSource, resetOnDisconnect,
			func(connectable ro.ConnectableObservable[int], inner *subscriberProbe) { inner.subscribe(connectable) })
	})
}

// reentrantConnectableSeeds generates the seeds of the two re-entrant connectable targets.
func reentrantConnectableSeeds(i int) []any {
	// items, trigger, yieldPattern, asyncSource, resetOnDisconnect
	return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%3 == 0}
}

// FuzzConnectableShareReset races subscribers against a hand-fed or self-driven source that completes or
// fails, for every combination of the Share reset flags. Each of the `subscribers` workers subscribes twice;
// the subscriptions of the first `leavingWorkers` workers unsubscribe right away.
//
// Invariant: contracts hold for every subscriber. When the reset flags say the next subscriber must
// get a fresh generation, it opens exactly one fresh source subscription and receives the fresh
// source's values.
//
// Seeds: all arguments spread over their range; the booleans alternate at different rates.
func FuzzConnectableShareReset(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, subscribers, leavingWorkers, yieldPattern, resetOnError, resetOnComplete, resetOnRefCountZero,
		// endSource, failAtEnd, selfDrivenSource, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0, i%4 < 2, i%5 < 3, i%7 < 3, i%11 < 5, i%13 < 6}
	})

	f.Fuzz(func(t *testing.T, items, subscribers, leavingWorkers, yieldPattern uint8, resetOnError, resetOnComplete, resetOnRefCountZero, endSource, failAtEnd, selfDrivenSource, asyncSource bool) {
		count := bounded(items, 0, maxItems)
		workerCount := bounded(subscribers, 1, maxWorkers)

		if selfDrivenSource {
			// A self-driven source always completes.
			endSource, failAtEnd = true, false
		}

		numbers := newControlledSource(count, selfDrivenSource, asyncSource, yieldPattern)
		shared := ro.ShareWithConfig(ro.ShareConfig[int]{
			Connector:           ro.NewPublishSubject[int],
			ResetOnError:        resetOnError,
			ResetOnComplete:     resetOnComplete,
			ResetOnRefCountZero: resetOnRefCountZero,
		})(numbers.observable())

		var wg sync.WaitGroup

		wg.Add(1)

		go func() {
			defer wg.Done()

			numbers.drive(count, yieldPattern, endSource, failAtEnd)
		}()

		probes, subscriptions := subscribeTwicePerWorker(&wg, shared, workerCount, bounded(leavingWorkers, 0, workerCount), yieldPattern)

		expectWaitGroupDone(t, "concurrent Share subscribers", &wg)

		for _, probe := range probes {
			probe.expectContract(t, "subscriber")
		}

		for _, subscription := range subscriptions {
			subscription.Unsubscribe()
		}

		// Without ResetOnRefCountZero, a subscriber arriving after the source ended opens a new generation
		// that legitimately stays subscribed, so nothing can be asserted. The same holds when the terminal
		// notification is kept instead of reset.
		keptTerminal := endSource && ((failAtEnd && !resetOnError) || (!failAtEnd && !resetOnComplete))
		if !resetOnRefCountZero || keptTerminal {
			return
		}

		scenario := fmt.Sprintf("resetOnError=%v resetOnComplete=%v endSource=%v failAtEnd=%v selfDriven=%v async=%v",
			resetOnError, resetOnComplete, endSource, failAtEnd, selfDrivenSource, asyncSource)

		// A reset generation has no owner left to unsubscribe its source, so the operator must have done it.
		// An asynchronous source may still be emitting when the last subscriber leaves, hence the wait.
		waitUntil(t, "source subscriptions to be released ("+scenario+")", func() bool { return numbers.counter.activeCount() == 0 })

		expectFreshGeneration(t, shared, numbers, count)
	})
}

// subscribeTwicePerWorker starts `workers` goroutines that subscribe twice to shared. The subscriptions of the
// first `leaving` workers unsubscribe right after subscribing, dropping the reference count while the source
// may still be emitting. The probes and subscriptions are indexed by worker*2+round.
func subscribeTwicePerWorker(wg *sync.WaitGroup, shared ro.Observable[int], workers, leaving int, yieldPattern uint8) ([]*subscriberProbe, []ro.Subscription) {
	probes := make([]*subscriberProbe, workers*2)
	subscriptions := make([]ro.Subscription, workers*2)

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)

		go func(worker int) {
			defer wg.Done()

			for round := 0; round < 2; round++ {
				yieldAt(int64(yieldPattern), worker*4+round)

				probe := &subscriberProbe{}
				subscription := probe.subscribe(shared)

				probes[worker*2+round] = probe
				subscriptions[worker*2+round] = subscription

				if worker < leaving {
					subscription.Unsubscribe()
				}
			}
		}(worker)
	}

	return probes, subscriptions
}

// expectFreshGeneration subscribes once more to a reset Share and checks that it opened exactly one fresh
// source subscription and received the fresh source's values.
func expectFreshGeneration(t *testing.T, shared ro.Observable[int], numbers *controlledSource, count int) {
	t.Helper()

	before := numbers.counter.totalCount()
	fresh := &subscriberProbe{}
	subscription := fresh.subscribe(shared)

	defer subscription.Unsubscribe()

	if got := numbers.counter.totalCount(); got != before+1 {
		t.Fatalf("next subscriber opened %d source subscriptions, want 1 fresh one", got-before)
	}

	if numbers.selfDriven != nil {
		waitUntil(t, "fresh subscriber to terminate", func() bool { return fresh.terminalCount() > 0 })

		if fresh.valueCount() != count {
			t.Fatalf("fresh subscriber received %d values from the fresh source, want %d", fresh.valueCount(), count)
		}

		return
	}

	numbers.emit(42)

	if fresh.valueCount() != 1 {
		t.Fatalf("fresh subscriber received %d values from the fresh source, want 1", fresh.valueCount())
	}
}

// FuzzConnectableShareReplayRefCount checks the documented contract of ShareReplay: the source is unsubscribed
// when every subscriber left. Each of the `subscribers` workers subscribes then unsubscribes; worker 0 also
// feeds a hand-fed source.
//
// Invariant: contracts hold for every subscriber, and no source subscription stays active once all
// subscribers left.
//
// Seeds: all arguments spread over their range; selfDrivenSource and asyncSource alternate.
func FuzzConnectableShareReplayRefCount(f *testing.F) {
	f.Skip("race: sharereplay-source-stays-subscribed; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, subscribers, replayBuffer, yieldPattern, selfDrivenSource, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, subscribers, replayBuffer, yieldPattern uint8, selfDrivenSource, asyncSource bool) {
		count := bounded(items, 0, maxItems)
		workerCount := bounded(subscribers, 1, maxWorkers)

		numbers := newControlledSource(count, selfDrivenSource, asyncSource, yieldPattern)
		shared := ro.ShareReplayWithConfig[int](bounded(replayBuffer, 1, replayBufferLimit), ro.ShareReplayConfig{ResetOnRefCountZero: false})(numbers.observable())

		probes := make([]*subscriberProbe, workerCount)

		var wg sync.WaitGroup

		for worker := range probes {
			probes[worker] = &subscriberProbe{}

			wg.Add(1)

			go func(worker int) {
				defer wg.Done()

				yieldAt(int64(yieldPattern), worker)

				subscription := probes[worker].subscribe(shared)

				if worker == 0 {
					for i := 0; i < count; i++ {
						numbers.emit(i)
					}
				}

				yieldAt(int64(yieldPattern), worker+maxWorkers)
				subscription.Unsubscribe()
			}(worker)
		}

		expectWaitGroupDone(t, "ShareReplay subscribers", &wg)

		for _, probe := range probes {
			probe.expectContract(t, "subscriber")
		}

		if active := numbers.counter.activeCount(); active != 0 {
			t.Fatalf("ShareReplay left %d source subscriptions active after all subscribers left", active)
		}
	})
}

// FuzzConnectableShareIsolation applies one Share operator to two different sources and races
// `subscribers` workers subscribing to both, then emits on both sources concurrently. Sources are hand-fed
// or self-driven.
//
// Invariant: state does not leak between the two applications. Subscribers of the first source only see its
// values, subscribers of the second only see values offset by secondSourceValueBase. Hand-fed
// subscribers see the whole sequence, and each hand-fed source is subscribed exactly once.
//
// Seeds: all arguments spread over their range; selfDrivenSource and asyncSource alternate.
func FuzzConnectableShareIsolation(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, subscribers, yieldPattern, selfDrivenSource, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, subscribers, yieldPattern uint8, selfDrivenSource, asyncSource bool) {
		count := bounded(items, 1, maxItems)
		workerCount := bounded(subscribers, 1, maxWorkers)

		share := ro.Share[int]()
		firstSource := newControlledSource(count, selfDrivenSource, asyncSource, yieldPattern)
		secondSource := newControlledSource(count, selfDrivenSource, asyncSource, yieldPattern+1)
		firstShared := share(firstSource.observable())
		secondShared := share(ro.Map(func(value int) int { return value + secondSourceValueBase })(secondSource.observable()))

		probes, subscriptions := subscribeToBothShares(t, firstShared, secondShared, workerCount, yieldPattern)

		emitOnBothSources(t, firstSource, secondSource, count, yieldPattern)

		expectEachSubscriberSawOnlyItsSource(t, probes, selfDrivenSource, count)

		for _, subscription := range subscriptions {
			subscription.Unsubscribe()
		}

		waitUntil(t, "sources to be released", func() bool {
			return firstSource.counter.activeCount() == 0 && secondSource.counter.activeCount() == 0
		})

		if !selfDrivenSource {
			if first, second := firstSource.counter.totalCount(), secondSource.counter.totalCount(); first != 1 || second != 1 {
				t.Fatalf("each source must be subscribed exactly once, got first=%d second=%d", first, second)
			}
		}
	})
}

// subscribeToBothShares subscribes `workers` probes to each share from concurrent goroutines. Even indexes
// subscribe to the first share, odd ones to the second.
func subscribeToBothShares(t *testing.T, firstShared, secondShared ro.Observable[int], workers int, yieldPattern uint8) ([]*subscriberProbe, []ro.Subscription) {
	t.Helper()

	probes := make([]*subscriberProbe, workers*2)
	subscriptions := make([]ro.Subscription, workers*2)

	var wg sync.WaitGroup

	for index := range probes {
		probes[index] = &subscriberProbe{}

		wg.Add(1)

		go func(index int) {
			defer wg.Done()

			yieldAt(int64(yieldPattern), index)

			if index%2 == 0 {
				subscriptions[index] = probes[index].subscribe(firstShared)
			} else {
				subscriptions[index] = probes[index].subscribe(secondShared)
			}
		}(index)
	}

	expectWaitGroupDone(t, "subscribing to both shares", &wg)

	return probes, subscriptions
}

// emitOnBothSources drives the two hand-fed sources from concurrent goroutines.
func emitOnBothSources(t *testing.T, firstSource, secondSource *controlledSource, count int, yieldPattern uint8) {
	t.Helper()

	var wg sync.WaitGroup

	wg.Add(2)

	go func() {
		defer wg.Done()

		firstSource.drive(count, yieldPattern, false, false)
	}()

	go func() {
		defer wg.Done()

		secondSource.drive(count, yieldPattern+1, false, false)
	}()

	expectWaitGroupDone(t, "emitting", &wg)
}

// expectEachSubscriberSawOnlyItsSource checks that even-indexed probes only saw values of the first source
// and odd-indexed ones only values of the second.
func expectEachSubscriberSawOnlyItsSource(t *testing.T, probes []*subscriberProbe, selfDriven bool, count int) {
	t.Helper()

	for index, probe := range probes {
		probe.expectContract(t, "subscriber")

		// Hand-fed subscribers were registered before any emission, so each sees the whole sequence;
		// a self-driven source may start before the later subscribers join.
		if !selfDriven && probe.valueCount() != count {
			t.Fatalf("subscriber %d received %d values, want %d", index, probe.valueCount(), count)
		}

		for _, value := range probe.received() {
			if (index%2 == 0) != (value < secondSourceValueBase) {
				t.Fatalf("subscriber %d (source %d) received foreign value %d", index, index%2, value)
			}
		}
	}
}

// FuzzConnectableConnectSubscribeLoop runs `bursts` rounds of Connect then Unsubscribe, and Subscribe then
// Unsubscribe, from many goroutines on one Connectable over a synchronous source.
//
// Invariant: nothing panics, deadlocks or leaks. Connect completes synchronously and its teardown
// resets the subject, racing with concurrent Connect and Subscribe calls.
//
// Seeds: bursts and yieldPattern spread over their range.
func FuzzConnectableConnectSubscribeLoop(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// bursts, yieldPattern
		return []any{seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, bursts, yieldPattern uint8) {
		burstCount := bounded(bursts, 1, maxLoopBursts)
		connectable := ro.Connectable(ro.Just(1, 2, 3))

		var wg sync.WaitGroup

		for worker := 0; worker < loopedSubscribers; worker++ {
			wg.Add(1)

			go func(worker int) {
				defer wg.Done()

				for burst := 0; burst < burstCount; burst++ {
					yieldAt(int64(yieldPattern), worker)
					connectable.Connect().Unsubscribe()
					yieldAt(int64(yieldPattern), worker+loopedSubscribers)
					connectable.Subscribe(ro.OnNext(func(int) {})).Unsubscribe()
				}
			}(worker)
		}

		expectWaitGroupDone(t, "Connect and Subscribe loops", &wg)
	})
}
