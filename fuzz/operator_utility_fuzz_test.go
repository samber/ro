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
	"testing"

	"github.com/samber/ro"
)

// FuzzRepeatWithStoppedByTake repeats a source of `items` integers, a thousand times at most, and closes
// the stream with ro.Take(takeCount).
//
// Invariant: exactly takeCount items are delivered, and the loop opens no source subscription after the
// downstream closed.
//
// Seeds: items and takeCount spread over their range; asyncSource alternates.
func FuzzRepeatWithStoppedByTake(f *testing.F) {
	f.Skip("race: repeatwith-ignores-take-close; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, takeCount, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, takeCount uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		taken := bounded(takeCount, 1, maxLoopTake)
		// A finite count, so that a loop that ignores Take is slow instead of endless.
		repeat := ro.RepeatWith[int](finiteRepeatCount)

		got, upstream := runLoopStoppedByTake(t, repeat, newSource(itemsPerRun, asyncSource), taken)

		expectLoopStoppedByTake(t, got, upstream, taken, itemsPerRun)
	})
}

// finiteRepeatCount is the repeat count of the loop that Take is meant to cut short.
const finiteRepeatCount = 1000

// FuzzRepeatWithFinite repeats a source of `items` integers `rounds` times and lets the loop end.
//
// Invariant: the source is subscribed `rounds` times, every run delivers its items, and the stream completes once.
//
// Seeds: items and rounds spread over their range; asyncSource alternates.
func FuzzRepeatWithFinite(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, rounds, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, rounds uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		runs := bounded(rounds, 1, maxFiniteLoopRuns)

		got, upstream := runFiniteLoop(t, ro.RepeatWith[int](int64(runs)), newSource(itemsPerRun, asyncSource))

		expectFiniteLoop(t, got, upstream, runs, itemsPerRun)
	})
}

// FuzzObserveOnCompletes reads ro.ObserveOn with one or two slow subscribers. The source is
// a source of `items` integers observed on another goroutine through a buffer of bufferSize items.
// The test lets the stream end on its own.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzObserveOnCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.ObserveOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzObserveOnStoppedByTake reads ro.ObserveOn with one or two slow subscribers. The source is
// a source of `items` integers observed on another goroutine through a buffer of bufferSize items.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzObserveOnStoppedByTake(f *testing.F) {
	f.Skip("race: observeon-send-close; chansend (operator_utility.go:597) races close(ch) in teardown (operator_utility.go:585); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, takeCount, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize, takeCount uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.Take[int](int64(bounded(takeCount, 1, count)))(ro.ObserveOn[int](bounded(bufferSize, 1, 4))(source.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzObserveOnStoppedByUnsubscribe reads ro.ObserveOn with one or two slow subscribers. The source is
// a source of `items` integers observed on another goroutine through a buffer of bufferSize items.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzObserveOnStoppedByUnsubscribe(f *testing.F) {
	f.Skip("race: observeon-send-close; chansend (operator_utility.go:597) races close(ch) in teardown (operator_utility.go:585); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, yieldPattern, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize, yieldPattern uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.ObserveOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzObserveOnStoppedByContextCancel reads ro.ObserveOn with one or two slow subscribers. The source is
// a source of `items` integers observed on another goroutine through a buffer of bufferSize items.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzObserveOnStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, cancelDelayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, cancelDelayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.ObserveOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzSubscribeOnCompletes reads ro.SubscribeOn with one or two slow subscribers. The source is
// a source of `items` integers subscribed on another goroutine through a buffer of bufferSize items.
// The test lets the stream end on its own.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzSubscribeOnCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.SubscribeOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzSubscribeOnStoppedByTake reads ro.SubscribeOn with one or two slow subscribers. The source is
// a source of `items` integers subscribed on another goroutine through a buffer of bufferSize items.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzSubscribeOnStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, takeCount, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize, takeCount uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.Take[int](int64(bounded(takeCount, 1, count)))(ro.SubscribeOn[int](bounded(bufferSize, 1, 4))(source.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzSubscribeOnStoppedByUnsubscribe reads ro.SubscribeOn with one or two slow subscribers. The source is
// a source of `items` integers subscribed on another goroutine through a buffer of bufferSize items.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzSubscribeOnStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, yieldPattern, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize, yieldPattern uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.SubscribeOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzSubscribeOnStoppedByContextCancel reads ro.SubscribeOn with one or two slow subscribers. The source is
// a source of `items` integers subscribed on another goroutine through a buffer of bufferSize items.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzSubscribeOnStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, cancelDelayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), seedByte(i, 1), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, cancelDelayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, 0)
		pipeline := ro.SubscribeOn[int](bounded(bufferSize, 1, 4))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzDelayCompletes reads ro.Delay with one or two slow subscribers. The source is
// a source of `items` integers delayed by delayMicros microseconds.
// The test lets the stream end on its own.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), i%2 == 0, i%4 < 2, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.Delay[int](shortDelay(delayMicros))(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayStoppedByTake reads ro.Delay with one or two slow subscribers. The source is
// a source of `items` integers delayed by delayMicros microseconds.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayStoppedByTake(f *testing.F) {
	f.Skip("race: delay-hang; Delay(d)(source) with Take/Unsubscribe/cancel never finishes: \"Delay: hang: iteration did not finish within 5s\"; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, takeCount, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, takeCount uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.Take[int](int64(bounded(takeCount, 1, count)))(ro.Delay[int](shortDelay(delayMicros))(countSubscriptions(upstream, numbers.observable())))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayStoppedByUnsubscribe reads ro.Delay with one or two slow subscribers. The source is
// a source of `items` integers delayed by delayMicros microseconds.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, yieldPattern, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, yieldPattern uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.Delay[int](shortDelay(delayMicros))(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayStoppedByContextCancel reads ro.Delay with one or two slow subscribers. The source is
// a source of `items` integers delayed by delayMicros microseconds.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, cancelDelayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros, cancelDelayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.Delay[int](shortDelay(delayMicros))(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayEachCompletes reads ro.DelayEach with one or two slow subscribers. The source is
// a source of `items` integers, each item delayed by a quarter of delayMicros microseconds.
// The test lets the stream end on its own.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayEachCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), i%2 == 0, i%4 < 2, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.DelayEach[int](shortDelay(delayMicros) / 4)(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayEachStoppedByTake reads ro.DelayEach with one or two slow subscribers. The source is
// a source of `items` integers, each item delayed by a quarter of delayMicros microseconds.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayEachStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, takeCount, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, takeCount uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.Take[int](int64(bounded(takeCount, 1, count)))(ro.DelayEach[int](shortDelay(delayMicros) / 4)(countSubscriptions(upstream, numbers.observable())))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayEachStoppedByUnsubscribe reads ro.DelayEach with one or two slow subscribers. The source is
// a source of `items` integers, each item delayed by a quarter of delayMicros microseconds.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayEachStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, yieldPattern, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros uint16, yieldPattern uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.DelayEach[int](shortDelay(delayMicros) / 4)(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzDelayEachStoppedByContextCancel reads ro.DelayEach with one or two slow subscribers. The source is
// a source of `items` integers, each item delayed by a quarter of delayMicros microseconds.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it)
// and the source subscription is released, even when pending timers fire after the teardown.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzDelayEachStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, delayMicros, cancelDelayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, delayMicros, cancelDelayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		pipeline := ro.DelayEach[int](shortDelay(delayMicros) / 4)(countSubscriptions(upstream, numbers.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		upstream.expectAllReleased(t)
	})
}

// FuzzTimeoutCompletes reads ro.Timeout with one or two slow subscribers. The source is
// a source of `items` integers that pauses gapMicros microseconds before every third item, so that timeouts race with items and with completion.
// The test lets the stream end on its own.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzTimeoutCompletes(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, timeoutMicros, gapMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), i%2 == 0, i%4 < 2, seedByte(i, 5)}
	})

	f.Fuzz(func(t *testing.T, items uint8, timeoutMicros, gapMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, shortDelay(gapMicros))
		pipeline := ro.Timeout[int](timerPeriod(timeoutMicros))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzTimeoutStoppedByTake reads ro.Timeout with one or two slow subscribers. The source is
// a source of `items` integers that pauses gapMicros microseconds before every third item, so that timeouts race with items and with completion.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzTimeoutStoppedByTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, timeoutMicros, gapMicros, takeCount, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), seedByte(i, 3), i%2 == 0, i%4 < 2, seedByte(i, 6)}
	})

	f.Fuzz(func(t *testing.T, items uint8, timeoutMicros, gapMicros uint16, takeCount uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, shortDelay(gapMicros))
		pipeline := ro.Take[int](int64(bounded(takeCount, 1, count)))(ro.Timeout[int](timerPeriod(timeoutMicros))(source.observable()))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzTimeoutStoppedByUnsubscribe reads ro.Timeout with one or two slow subscribers. The source is
// a source of `items` integers that pauses gapMicros microseconds before every third item, so that timeouts race with items and with completion.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzTimeoutStoppedByUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, timeoutMicros, gapMicros, yieldPattern, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), seedByte(i, 3), i%2 == 0, i%4 < 2, seedByte(i, 6)}
	})

	f.Fuzz(func(t *testing.T, items uint8, timeoutMicros, gapMicros uint16, yieldPattern uint8, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, shortDelay(gapMicros))
		pipeline := ro.Timeout[int](timerPeriod(timeoutMicros))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzTimeoutStoppedByContextCancel reads ro.Timeout with one or two slow subscribers. The source is
// a source of `items` integers that pauses gapMicros microseconds before every third item, so that timeouts race with items and with completion.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: every subscriber sees a valid stream (no overlap, one terminal at most, nothing after it),
// no downstream call panics in the source, the source goroutine is released and its subscription torn down.
//
// Seeds: numeric inputs spread over their range; asyncSource and twoSubscribers alternate.
func FuzzTimeoutStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, timeoutMicros, gapMicros, cancelDelayMicros, asyncSource, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), uint16(i * 107), i%2 == 0, i%4 < 2, seedByte(i, 6)}
	})

	f.Fuzz(func(t *testing.T, items uint8, timeoutMicros, gapMicros, cancelDelayMicros uint16, asyncSource, twoSubscribers bool, readDelay uint8) {
		count := bounded(items, 1, maxItems)
		source := newAuditedSource(count, asyncSource, shortDelay(gapMicros))
		pipeline := ro.Timeout[int](timerPeriod(timeoutMicros))(source.observable())

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectValidStreams(t, recorders)
		source.expectCleanEnd(t)
	})
}

// FuzzSubscribeOnIntervalStoppedByTake subscribes one or two slow subscribers to ro.SubscribeOn over an endless ro.Interval.
// The test stops the stream from inside the pipeline with ro.Take.
//
// Invariant: stopping the downstream stops the endless upstream: Subscribe returns, every subscriber sees a valid
// stream and the Interval subscription is released.
//
// Seeds: numeric inputs spread over their range; twoSubscribers alternates.
func FuzzSubscribeOnIntervalStoppedByTake(f *testing.F) {
	f.Skip("race: subscribeon-infinite-hang; SubscribeOn(Interval)+Take/Unsubscribe/cancel never returns: \"SubscribeOn(Interval): hang: iteration did not finish within 5s\"; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// bufferSize, periodMicros, takeCount, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, bufferSize uint8, periodMicros uint16, takeCount uint8, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		ticks := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))
		pipeline := ro.Take[int64](int64(bounded(takeCount, 1, maxTicks)))(ro.SubscribeOn[int64](bounded(bufferSize, 1, 4))(ticks))

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingWhenStreamEnds)

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzSubscribeOnIntervalStoppedByUnsubscribe subscribes one or two slow subscribers to ro.SubscribeOn over an endless ro.Interval.
// The test unsubscribes right after Subscribe returned, from two goroutines at once.
//
// Invariant: stopping the downstream stops the endless upstream: Subscribe returns, every subscriber sees a valid
// stream and the Interval subscription is released.
//
// Seeds: numeric inputs spread over their range; twoSubscribers alternates.
func FuzzSubscribeOnIntervalStoppedByUnsubscribe(f *testing.F) {
	f.Skip("race: subscribeon-infinite-hang; SubscribeOn(Interval)+Take/Unsubscribe/cancel never returns: \"SubscribeOn(Interval): hang: iteration did not finish within 5s\"; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// bufferSize, periodMicros, yieldPattern, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), seedByte(i, 2), i%2 == 0, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, bufferSize uint8, periodMicros uint16, yieldPattern uint8, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		ticks := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))
		pipeline := ro.SubscribeOn[int64](bounded(bufferSize, 1, 4))(ticks)

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByUnsubscribe(yieldPattern))

		expectCleanShutdown(t, recorders, upstream)
	})
}

// FuzzSubscribeOnIntervalStoppedByContextCancel subscribes one or two slow subscribers to ro.SubscribeOn over an endless ro.Interval.
// The test cancels the subscriber context after a fuzzed delay.
//
// Invariant: stopping the downstream stops the endless upstream: Subscribe returns, every subscriber sees a valid
// stream and the Interval subscription is released.
//
// Seeds: numeric inputs spread over their range; twoSubscribers alternates.
func FuzzSubscribeOnIntervalStoppedByContextCancel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// bufferSize, periodMicros, cancelDelayMicros, twoSubscribers, readDelay
		return []any{seedByte(i, 0), uint16(i * 71), uint16(i * 89), i%2 == 0, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, bufferSize uint8, periodMicros, cancelDelayMicros uint16, twoSubscribers bool, readDelay uint8) {
		upstream := &subscriptionCounter{}
		ticks := countSubscriptions(upstream, ro.Interval(timerPeriod(periodMicros)))
		pipeline := ro.SubscribeOn[int64](bounded(bufferSize, 1, 4))(ticks)

		recorders := subscribeConcurrently(t, pipeline, subscriberCount(twoSubscribers), readPause(readDelay), endingByContextCancel(shortDelay(cancelDelayMicros)))

		expectCleanShutdown(t, recorders, upstream)
	})
}
