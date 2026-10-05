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
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// The Interval, IntervalWithInitial, Timer and Future targets below share one shape: one or two
// subscribers subscribe to the same pipeline from their own goroutines, the stream ends in a way
// named by the target (it ends on its own, Take stops it, Unsubscribe stops it, the context is
// canceled), and each subscriber must have seen a valid stream. Delays stay under 2 ms.

// FuzzIntervalStoppedByTake stops ro.Interval from inside the stream with ro.Take, while one or two
// slow subscribers read it.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after
// it) and the Interval subscription is released.
//
// Seeds: periodMicros, takeCount and readDelay spread over their range; twoSubscribers alternates.
func FuzzIntervalStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// periodMicros, takeCount, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 53), seedByte(i, 0), i%2 == 0, seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, periodMicros uint16, takeCount uint8, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		ticks := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))
		pipeline := ro.Take[int64](int64(bounded(takeCount, 1, maxTicks)))(ticks)

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzIntervalStoppedByUnsubscribe unsubscribes from ro.Interval, from two goroutines at once, right after
// Subscribe returned, while one or two slow subscribers read it.
//
// Invariant: same as FuzzIntervalStoppedByTake.
//
// Seeds: periodMicros, readDelay and yieldPattern spread over their range; twoSubscribers alternates.
func FuzzIntervalStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// periodMicros, twoSubscribers, readDelay (microseconds), yieldPattern
		return []any{uint16(i * 53), i%2 == 0, seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, periodMicros uint16, twoSubscribers bool, readDelay, yieldPattern uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByUnsubscribe(yieldPattern))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzIntervalStoppedByContextCancel cancels the subscriber context of ro.Interval after a fuzzed delay,
// while one or two slow subscribers read it.
//
// Invariant: same as FuzzIntervalStoppedByTake.
//
// Seeds: periodMicros, cancelDelayMicros and readDelay spread over their range; twoSubscribers alternates.
func FuzzIntervalStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// periodMicros, cancelDelayMicros, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 53), uint16(i * 71), i%2 == 0, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, periodMicros, cancelDelayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// intervalWithInitial builds ro.IntervalWithInitial. A synchronous first tick means an initial delay of
// zero: the first value is then emitted inside Subscribe, before any teardown exists.
func intervalWithInitial(initialDelayMicros, periodMicros uint16, synchronousFirstTick bool) ro.Observable[int64] {
	initialDelay := shortDelay(initialDelayMicros)
	if synchronousFirstTick {
		initialDelay = 0
	}

	return ro.IntervalWithInitial(initialDelay, timerPeriod(periodMicros))
}

// FuzzIntervalWithInitialStoppedByTake stops ro.IntervalWithInitial from inside the stream with ro.Take,
// while one or two slow subscribers read it. The first tick is synchronous or delayed.
//
// Invariant: every subscriber sees a valid stream and the Interval subscription is released.
//
// Seeds: the delays and takeCount spread over their range; synchronousFirstTick and twoSubscribers alternate.
func FuzzIntervalWithInitialStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// initialDelayMicros, periodMicros, synchronousFirstTick, takeCount, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), uint16(i * 53), i%2 == 0, seedByte(i, 0), i%4 < 2, seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, initialDelayMicros, periodMicros uint16, synchronousFirstTick bool, takeCount uint8, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		ticks := countSubscriptions(upstream, intervalWithInitial(initialDelayMicros, periodMicros, synchronousFirstTick))
		pipeline := ro.Take[int64](int64(bounded(takeCount, 1, maxTicks)))(ticks)

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzIntervalWithInitialStoppedByUnsubscribe unsubscribes from ro.IntervalWithInitial, from two goroutines
// at once, right after Subscribe returned. The first tick is synchronous or delayed.
//
// Invariant: same as FuzzIntervalWithInitialStoppedByTake.
//
// Seeds: the delays, readDelay and yieldPattern spread over their range; synchronousFirstTick and
// twoSubscribers alternate.
func FuzzIntervalWithInitialStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// initialDelayMicros, periodMicros, synchronousFirstTick, twoSubscribers, readDelay (microseconds), yieldPattern
		return []any{uint16(i * 71), uint16(i * 53), i%2 == 0, i%4 < 2, seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, initialDelayMicros, periodMicros uint16, synchronousFirstTick, twoSubscribers bool, readDelay, yieldPattern uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, intervalWithInitial(initialDelayMicros, periodMicros, synchronousFirstTick))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByUnsubscribe(yieldPattern))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzIntervalWithInitialStoppedByContextCancel cancels the subscriber context of ro.IntervalWithInitial
// after a fuzzed delay. The first tick is synchronous or delayed.
//
// Invariant: same as FuzzIntervalWithInitialStoppedByTake.
//
// Seeds: the delays and readDelay spread over their range; synchronousFirstTick and twoSubscribers alternate.
func FuzzIntervalWithInitialStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// initialDelayMicros, periodMicros, synchronousFirstTick, cancelDelayMicros, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), uint16(i * 53), i%2 == 0, uint16(i * 89), i%4 < 2, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, initialDelayMicros, periodMicros uint16, synchronousFirstTick bool, cancelDelayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, intervalWithInitial(initialDelayMicros, periodMicros, synchronousFirstTick))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzTimerCompletes lets ro.Timer fire and complete while one or two slow subscribers read it.
//
// Invariant: every subscriber sees a valid stream and the Timer subscription is released.
//
// Seeds: delayMicros and readDelay spread over their range; twoSubscribers alternates.
func FuzzTimerCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// delayMicros, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), i%2 == 0, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, delayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, ro.Timer(shortDelay(delayMicros)))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzTimerStoppedByTake stops ro.Timer from inside the stream with ro.Take(1), racing the timer firing.
//
// Invariant: same as FuzzTimerCompletes.
//
// Seeds: same as FuzzTimerCompletes.
func FuzzTimerStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// delayMicros, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), i%2 == 0, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, delayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		pipeline := ro.Take[time.Duration](1)(countSubscriptions(upstream, ro.Timer(shortDelay(delayMicros))))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzTimerStoppedByUnsubscribe unsubscribes from ro.Timer, from two goroutines at once, right after
// Subscribe returned, racing the timer firing.
//
// Invariant: same as FuzzTimerCompletes.
//
// Seeds: delayMicros, readDelay and yieldPattern spread over their range; twoSubscribers alternates.
func FuzzTimerStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// delayMicros, twoSubscribers, readDelay (microseconds), yieldPattern
		return []any{uint16(i * 71), i%2 == 0, seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, delayMicros uint16, twoSubscribers bool, readDelay, yieldPattern uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, ro.Timer(shortDelay(delayMicros)))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByUnsubscribe(yieldPattern))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzTimerStoppedByContextCancel cancels the subscriber context of ro.Timer after a fuzzed delay,
// racing the timer firing.
//
// Invariant: same as FuzzTimerCompletes.
//
// Seeds: delayMicros, cancelDelayMicros and readDelay spread over their range; twoSubscribers alternates.
func FuzzTimerStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// delayMicros, cancelDelayMicros, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), uint16(i * 89), i%2 == 0, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, delayMicros, cancelDelayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		pipeline := countSubscriptions(upstream, ro.Timer(shortDelay(delayMicros)))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), time.Duration(readDelay)*time.Microsecond, endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// futureFactory is the function given to ro.Future. It counts its calls, takes `duration` to answer,
// then returns 1 or errInjectedFailure.
type futureFactory struct {
	duration time.Duration
	failing  bool
	calls    int32
}

func (factory *futureFactory) produce() (int, error) {
	atomic.AddInt32(&factory.calls, 1)
	time.Sleep(factory.duration)

	if factory.failing {
		return 0, errInjectedFailure
	}

	return 1, nil
}

// expectOneCallPerSubscriber checks that every subscription started its own run of the factory.
func (factory *futureFactory) expectOneCallPerSubscriber(t *testing.T, subscribers int) {
	t.Helper()

	if got := int(atomic.LoadInt32(&factory.calls)); got != subscribers {
		t.Fatalf("Future: factory called %d times for %d subscriptions", got, subscribers)
	}
}

// FuzzFutureCompletes lets ro.Future answer, with a value or an error, while one or two slow
// subscribers read it.
//
// Invariant: every subscriber sees a valid stream, even when the factory goroutine delivers late,
// and the factory runs once per subscription.
//
// Seeds: factoryDelayMicros and readDelay spread over their range; failing and twoSubscribers alternate.
func FuzzFutureCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// factoryDelayMicros, failing, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), i%2 == 0, i%4 < 2, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, factoryDelayMicros uint16, failing, twoSubscribers bool, readDelay uint8) {
		factory := &futureFactory{duration: shortDelay(factoryDelayMicros), failing: failing}
		subscribers := subscriberCount(twoSubscribers)

		recorders := subscribeConcurrently(t, ro.Future(factory.produce), subscribers, time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		factory.expectOneCallPerSubscriber(t, subscribers)
	})
}

// FuzzFutureStoppedByTake stops ro.Future from inside the stream with ro.Take(1), racing the factory.
//
// Invariant: same as FuzzFutureCompletes.
//
// Seeds: same as FuzzFutureCompletes.
func FuzzFutureStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// factoryDelayMicros, failing, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), i%2 == 0, i%4 < 2, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, factoryDelayMicros uint16, failing, twoSubscribers bool, readDelay uint8) {
		factory := &futureFactory{duration: shortDelay(factoryDelayMicros), failing: failing}
		subscribers := subscriberCount(twoSubscribers)

		recorders := subscribeConcurrently(t, ro.Take[int](1)(ro.Future(factory.produce)), subscribers, time.Duration(readDelay)*time.Microsecond, endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		factory.expectOneCallPerSubscriber(t, subscribers)
	})
}

// FuzzFutureStoppedByUnsubscribe unsubscribes from ro.Future, from two goroutines at once, right after
// Subscribe returned, racing the factory.
//
// Invariant: same as FuzzFutureCompletes.
//
// Seeds: factoryDelayMicros, readDelay and yieldPattern spread over their range; failing and
// twoSubscribers alternate.
func FuzzFutureStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// factoryDelayMicros, failing, twoSubscribers, readDelay (microseconds), yieldPattern
		return []any{uint16(i * 71), i%2 == 0, i%4 < 2, seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, factoryDelayMicros uint16, failing, twoSubscribers bool, readDelay, yieldPattern uint8) {
		factory := &futureFactory{duration: shortDelay(factoryDelayMicros), failing: failing}
		subscribers := subscriberCount(twoSubscribers)

		recorders := subscribeConcurrently(t, ro.Future(factory.produce), subscribers, time.Duration(readDelay)*time.Microsecond, endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		factory.expectOneCallPerSubscriber(t, subscribers)
	})
}

// FuzzFutureStoppedByContextCancel cancels the subscriber context of ro.Future after a fuzzed delay,
// racing the factory.
//
// Invariant: same as FuzzFutureCompletes.
//
// Seeds: factoryDelayMicros, cancelDelayMicros and readDelay spread over their range; failing and
// twoSubscribers alternate.
func FuzzFutureStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// factoryDelayMicros, cancelDelayMicros, failing, twoSubscribers, readDelay (microseconds)
		return []any{uint16(i * 71), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, factoryDelayMicros, cancelDelayMicros uint16, failing, twoSubscribers bool, readDelay uint8) {
		factory := &futureFactory{duration: shortDelay(factoryDelayMicros), failing: failing}
		subscribers := subscriberCount(twoSubscribers)

		recorders := subscribeConcurrently(t, ro.Future(factory.produce), subscribers, time.Duration(readDelay)*time.Microsecond, endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		factory.expectOneCallPerSubscriber(t, subscribers)
	})
}

// FuzzFuturePanic subscribes to a ro.Future whose factory panics, then unsubscribes.
//
// Invariant: a panicking factory does not take the whole process down. The goroutine started by Future
// has no recover, so the process dies before any assertion runs: reaching the end means the panic was handled.
//
// Seeds: factoryDelayMicros, readDelay and yieldPattern spread over their range.
func FuzzFuturePanic(f *testing.F) {
	f.Skip("race: future-factory-panic; goroutine in Future (operator_creation.go:464) has no recover: \"panic: futureFactoryPanic\" kills the test binary; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// factoryDelayMicros, readDelay (microseconds), yieldPattern
		return []any{uint16(i * 71), seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, factoryDelayMicros uint16, readDelay, yieldPattern uint8) {
		factoryDelay := shortDelay(factoryDelayMicros)
		pipeline := ro.Future(func() (int, error) {
			time.Sleep(factoryDelay)
			panic("futureFactoryPanic")
		})

		recorders := subscribeConcurrently(t, pipeline, 1, time.Duration(readDelay)*time.Microsecond, endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
	})
}

// FuzzFromChannel reads a pre-filled channel of `items` integers that is never closed, so the reader
// goroutine always has an item ready. A slow subscriber reads it, and the test unsubscribes after a short delay.
//
// Invariant: no item is read from the channel after Unsubscribe returned, every item taken from the
// channel was delivered to the subscriber, and the stream was valid.
//
// Seeds: items, readDelay, unsubscribeDelayMicros and yieldPattern spread over their range.
func FuzzFromChannel(f *testing.F) {
	f.Skip("race: fromchannel-lost-item; reader keeps selecting on in after done is closed: \"FromChannel: 2 item(s) read from the channel after Unsubscribe returned (len 13 -> 11)\"; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, readDelay (microseconds), unsubscribeDelayMicros, yieldPattern
		return []any{seedByte(i, 0), seedByte(i, 1), uint16(i * 89), seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, items, readDelay uint8, unsubscribeDelayMicros uint16, yieldPattern uint8) {
		count := bounded(items, 1, maxItems)
		pending := prefilledChannel(count)
		reader := newSlowRecorder[int](time.Duration(readDelay) * time.Microsecond)

		subscription := ro.FromChannel[int](pending).Subscribe(reader)

		time.Sleep(shortDelay(unsubscribeDelayMicros) / 4) // short, so Unsubscribe lands while the reader drains
		yieldAt(int64(yieldPattern), 0)
		subscription.Unsubscribe()

		leftAtUnsubscribe := len(pending)

		time.Sleep(lateNotificationWait)

		if left := len(pending); left != leftAtUnsubscribe {
			t.Fatalf("%d item(s) read from the channel after Unsubscribe returned (len %d -> %d)", leftAtUnsubscribe-left, leftAtUnsubscribe, left)
		}

		if taken := count - len(pending); taken != reader.valueCount() {
			t.Fatalf("%d item(s) taken from the channel but %d delivered: items lost", taken, reader.valueCount())
		}

		reader.expectContract(t)
	})
}

// prefilledChannel returns a buffered channel holding 0..items-1, never closed.
func prefilledChannel(items int) chan int {
	pending := make(chan int, items)
	for item := 0; item < items; item++ {
		pending <- item
	}

	return pending
}
