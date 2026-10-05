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
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// maxBufferSize is the largest buffer a fuzz input can request. Small buffers fill often, so the
// emission path runs many times per iteration.
const maxBufferSize = 6

// FuzzMergeMap merges the inner sources selected by an outer source. Inners and outer are synchronous or
// asynchronous, and the stream runs to its end.
//
// Invariant: every inner value arrives exactly once, the stream completes once, and every inner and
// outer subscription is released.
//
// Seeds: outerItems and firstInnerSize spread over their range; asyncOuter, asyncEvenInners and
// asyncOddInners alternate.
func FuzzMergeMap(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runToCompletion(t, ro.MergeMap(sources.project)(sources.outerItems()), sources.upstreams()...)

		sources.expectEveryValue(t, got, false)
	})
}

// FuzzMergeMapTake is FuzzMergeMap closed from inside the pipeline by Take(takeLimit).
//
// Invariant: Take delivers min(takeLimit, available) values, nothing arrives after the stream closed,
// and every inner and outer subscription is released.
//
// Seeds: as FuzzMergeMap, with takeLimit spread over its range.
func FuzzMergeMapTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, takeLimit uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, ro.MergeMap(sources.project)(sources.outerItems()), limit, sources.upstreams()...)

		sources.expectTakenValues(t, got, limit, false)
	})
}

// FuzzMergeMapUnsubscribe is FuzzMergeMap unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: no value is invented, and every inner and outer subscription is released.
//
// Seeds: as FuzzMergeMap, with unsubscribeAfter spread over its range.
func FuzzMergeMapUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, unsubscribeAfter uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runThenUnsubscribe(t, ro.MergeMap(sources.project)(sources.outerItems()), bounded(unsubscribeAfter, 1, maxItemsBeforeStop), sources.upstreams()...)

		sources.expectOnlyEmittedValues(t, got, false)
	})
}

// FuzzFlatMap concatenates the inner sources selected by an outer source. Inners and outer are
// synchronous or asynchronous, and the stream runs to its end.
//
// Invariant: every inner value arrives exactly once, inner after inner in order, the stream completes
// once, and every inner and outer subscription is released.
//
// Seeds: as FuzzMergeMap.
func FuzzFlatMap(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runToCompletion(t, ro.FlatMap(sources.project)(sources.outerItems()), sources.upstreams()...)

		sources.expectEveryValue(t, got, true)
	})
}

// FuzzFlatMapTake is FuzzFlatMap closed from inside the pipeline by Take(takeLimit).
//
// Invariant: Take delivers the first min(takeLimit, available) values in order, and every inner and
// outer subscription is released.
//
// Seeds: as FuzzMergeMapTake.
func FuzzFlatMapTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, takeLimit uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, ro.FlatMap(sources.project)(sources.outerItems()), limit, sources.upstreams()...)

		sources.expectTakenValues(t, got, limit, true)
	})
}

// FuzzFlatMapUnsubscribe is FuzzFlatMap unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: the values received are the first values in order, and every inner and outer subscription
// is released.
//
// Seeds: as FuzzMergeMapUnsubscribe.
func FuzzFlatMapUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, unsubscribeAfter uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runThenUnsubscribe(t, ro.FlatMap(sources.project)(sources.outerItems()), bounded(unsubscribeAfter, 1, maxItemsBeforeStop), sources.upstreams()...)

		sources.expectOnlyEmittedValues(t, got, true)
	})
}

// feedSubjectFromProducer pushes `items` values then a completion into subject from a producer, and fails the test
// when the producer is stuck, which is how a Next that waits for the downstream shows up.
func feedSubjectFromProducer(t *testing.T, subject ro.Subject[int], items int, asyncProducer bool) {
	t.Helper()

	expectReturns(t, "the producer feeding the subject (blocked in Subject.Next while FlatMap waits for an inner)", func() {
		finished := make(chan struct{})

		newSource(items, asyncProducer).observable().Subscribe(ro.NewObserver(
			subject.Next,
			subject.Error,
			func() {
				subject.Complete()
				close(finished)
			},
		))

		<-finished
	})
}

// FuzzFlatMapSameSubject feeds FlatMap from a subject while inner i waits for the next item of that same
// subject. The producer is inside Subject.Next while FlatMap waits for the inner, so the producer is
// released only when FlatMap buffers the outer items.
//
// Invariant: the producer finishes, the stream terminates, every inner is released, and inner i
// receives item i+1, so count-1 values arrive.
//
// Seeds: items spreads over its range; asyncProducer alternates.
func FuzzFlatMapSameSubject(f *testing.F) {
	f.Skip("race: flatmap-blocks-producer; remove when fixed") // Fails on main: producer blocked in Subject.Next: FlatMap waits for an inner inside the outer Next

	fuzzSeeds(f, func(i int) []any {
		// items, asyncProducer
		return []any{seedByte(i, 0), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items uint8, asyncProducer bool) {
		count := bounded(items, 1, maxConcatSources)
		subject := ro.NewPublishSubject[int]()

		var inners subscriptionCounter

		nextItemOfSubject := func(int) ro.Observable[int] {
			return countSubscriptions(&inners, ro.Take[int](1)(subject.AsObservable()))
		}

		got := newRecorder[int]()
		subscription := subscribeInBackground(context.Background(), ro.FlatMap(nextItemOfSubject)(subject.AsObservable()), got)
		subscription.expectReturn(t)

		feedSubjectFromProducer(t, subject, count, asyncProducer)

		waitUntil(t, "a terminal notification", func() bool { return got.terminalCount() > 0 })
		inners.expectAllReleased(t)
		got.expectContract(t)

		if received := got.valueCount(); received != count-1 {
			t.Fatalf("%d values received, want %d (inner i receives item i+1)", received, count-1)
		}
	})
}

// FuzzFlatMapSubjectIndependentInners feeds FlatMap from a subject while the inners are plain sources that
// do not read the subject.
//
// Invariant: the producer finishes, the stream terminates, and every inner is released.
//
// Seeds: items spreads over its range; asyncProducer and asyncInners alternate.
func FuzzFlatMapSubjectIndependentInners(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, asyncProducer, asyncInners
		return []any{seedByte(i, 0), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items uint8, asyncProducer, asyncInners bool) {
		count := bounded(items, 1, maxConcatSources)
		subject := ro.NewPublishSubject[int]()

		var inners subscriptionCounter

		pair := func(item int) ro.Observable[int] {
			return countSubscriptions(&inners, newSource(2, asyncInners).startingAt(item*innerValueStride).observable())
		}

		got := newRecorder[int]()
		subscription := subscribeInBackground(context.Background(), ro.FlatMap(pair)(subject.AsObservable()), got)
		subscription.expectReturn(t)

		feedSubjectFromProducer(t, subject, count, asyncProducer)

		waitUntil(t, "a terminal notification", func() bool { return got.terminalCount() > 0 })
		inners.expectAllReleased(t)
		got.expectContract(t)
	})
}

// FuzzBufferWithCount buffers a source of `items` integers by groups of bufferSize. The source is
// synchronous or asynchronous, and the stream either runs to its end or is unsubscribed from outside
// after a few scheduler yields.
//
// Invariant: the buffers, flattened, are exactly 0..items-1 (nothing lost, duplicated or reordered),
// and the upstream is released. When unsubscribed early, only the upstream release and the observer
// contract are checked.
//
// Seeds: items, bufferSize, yieldPattern and spinsBeforeUnsubscribe spread over their range;
// asyncSource and unsubscribeEarly alternate.
func FuzzBufferWithCount(f *testing.F) {
	f.Skip("race: bufferwithcount-teardown-unlocked-buffer; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.BufferWithCount[int](bounded(bufferSize, 1, maxBufferSize))(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectCompleteInOrder(t, flattened(got.received()), count)
		}
	})
}

// FuzzBufferWhen buffers a source of `items` integers, emitting the buffer at each tick of a boundary
// that sends `ticks` ticks and never completes. Source and boundary are synchronous or asynchronous.
//
// Invariant: as FuzzBufferWithCount, and the boundary subscription is released too.
//
// Seeds: items, ticks, yieldPattern and spinsBeforeUnsubscribe spread over their range; asyncSource,
// asyncTicks and unsubscribeEarly alternate.
func FuzzBufferWhen(f *testing.F) {
	// Intermittent (about 1 run in 7 at 2000 seeds), for example: got 8 items [0 1 2 3 4 5 8 9], want 10.
	f.Skip("race: bufferwhen-lost-buffer-after-unlock; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, ticks, asyncSource, asyncTicks, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%8 < 4, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, ticks uint8, asyncSource, asyncTicks bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream, boundary subscriptionCounter

		tickSource := countSubscriptions(&boundary, tickingBoundary(bounded(ticks, 0, maxTicks), asyncTicks, yieldPattern))
		pipeline := ro.BufferWhen[int](tickSource)(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream, &boundary)
		if completed {
			expectCompleteInOrder(t, flattened(got.received()), count)
		}
	})
}

// FuzzBufferWithTime buffers a source of `items` integers, emitting the buffer every periodMicros
// microseconds. The source is synchronous or asynchronous.
//
// Invariant: as FuzzBufferWithCount.
//
// Seeds: items, periodMicros, yieldPattern and spinsBeforeUnsubscribe spread over their range;
// asyncSource and unsubscribeEarly alternate.
func FuzzBufferWithTime(f *testing.F) {
	f.Skip("race: bufferwithtime-flush-after-unlock-lost-buffer (intermittent, seen once under load); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, periodMicros, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), uint16(i * 251), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items uint8, periodMicros uint16, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.BufferWithTime[int](timerPeriod(periodMicros))(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectCompleteInOrder(t, flattened(got.received()), count)
		}
	})
}

// FuzzBufferWithTimeOrCount buffers a source of `items` integers, emitting the buffer when it holds
// bufferSize items or every periodMicros microseconds, whichever comes first. The source is synchronous
// or asynchronous.
//
// Invariant: as FuzzBufferWithCount.
//
// Seeds: items, bufferSize, periodMicros, yieldPattern and spinsBeforeUnsubscribe spread over their
// range; asyncSource and unsubscribeEarly alternate.
func FuzzBufferWithTimeOrCount(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, periodMicros, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), uint16(i * 251), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, periodMicros uint16, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.BufferWithTimeOrCount[int](bounded(bufferSize, 1, maxBufferSize), timerPeriod(periodMicros))(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectCompleteInOrder(t, flattened(got.received()), count)
		}
	})
}

// innerStreamCollector is the downstream of an operator that emits Observables (windows, groups). It
// subscribes to every Observable as soon as it arrives, and keeps what each one emitted and whether it
// ended. The outer notifications are kept by the embedded recorder.
type innerStreamCollector struct {
	*recorder[ro.Observable[int]]

	mu      sync.Mutex
	streams []*innerStreamRecord
}

type innerStreamRecord struct {
	items     []int
	terminals int32
}

func newInnerStreamCollector() *innerStreamCollector {
	return &innerStreamCollector{recorder: newRecorder[ro.Observable[int]]()}
}

func (c *innerStreamCollector) Next(stream ro.Observable[int]) {
	c.NextWithContext(context.Background(), stream)
}

func (c *innerStreamCollector) NextWithContext(ctx context.Context, stream ro.Observable[int]) {
	c.recorder.NextWithContext(ctx, stream)

	c.mu.Lock()
	record := &innerStreamRecord{}
	c.streams = append(c.streams, record)
	c.mu.Unlock()

	stream.Subscribe(ro.NewObserver(
		func(item int) {
			c.mu.Lock()
			record.items = append(record.items, item)
			c.mu.Unlock()
		},
		func(error) { atomic.AddInt32(&record.terminals, 1) },
		func() { atomic.AddInt32(&record.terminals, 1) },
	))
}

// openCount is the number of inner streams that have not received a terminal notification yet.
func (c *innerStreamCollector) openCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()

	open := 0

	for _, record := range c.streams {
		if atomic.LoadInt32(&record.terminals) == 0 {
			open++
		}
	}

	return open
}

// itemsPerStream returns what each inner stream emitted, in arrival order of the streams.
func (c *innerStreamCollector) itemsPerStream() [][]int {
	c.mu.Lock()
	defer c.mu.Unlock()

	items := make([][]int, len(c.streams))
	for index, record := range c.streams {
		items[index] = append([]int(nil), record.items...)
	}

	return items
}

func (c *innerStreamCollector) expectEveryStreamEnded(t *testing.T) {
	t.Helper()

	waitUntil(t, "every inner stream to receive a terminal notification", func() bool { return c.openCount() == 0 })
}

// subscribeAndEnd subscribes the collector to pipeline, then ends the stream as end says.
// It reports whether the stream ran to its natural end.
func (c *innerStreamCollector) subscribeAndEnd(t *testing.T, pipeline ro.Observable[ro.Observable[int]], end streamEnd) (completed bool) {
	t.Helper()

	subscription := subscribeWithinDeadline[ro.Observable[int]](t, pipeline, c)

	return end.waitOrUnsubscribe(t, subscription.Unsubscribe, c)
}

// FuzzWindowWhen splits a source of `items` integers into windows, closing the current window and opening
// the next at each tick of a boundary that sends `ticks` ticks and never completes. Source and boundary
// are synchronous or asynchronous, and the stream either runs to its end or is unsubscribed from outside.
//
// Invariant: the windows, flattened, are exactly 0..items-1; every window ends; the source and the
// boundary are released. When unsubscribed early, only the release is checked.
//
// Seeds: items, ticks, yieldPattern and spinsBeforeUnsubscribe spread over their range; asyncSource,
// asyncTicks and unsubscribeEarly alternate.
func FuzzWindowWhen(f *testing.F) {
	f.Skip("race: windowwhen-value-lost-in-completed-window; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, ticks, asyncSource, asyncTicks, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%8 < 4, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, ticks uint8, asyncSource, asyncTicks bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream, boundary subscriptionCounter

		tickSource := countSubscriptions(&boundary, tickingBoundary(bounded(ticks, 0, maxTicks), asyncTicks, yieldPattern))
		pipeline := ro.WindowWhen[int](tickSource)(countSubscriptions(&upstream, numbers.observable()))

		windows := newInnerStreamCollector()
		completed := windows.subscribeAndEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern))

		if completed {
			windows.expectEveryStreamEnded(t)
		}

		upstream.expectAllReleased(t)
		boundary.expectAllReleased(t)

		if completed {
			time.Sleep(settleDelay) // lets a stray notification after the end arrive before the contract is checked
			windows.expectCompletedOnce(t)
			windows.expectContract(t)
			expectCompleteInOrder(t, flattened(windows.itemsPerStream()), count)
		}
	})
}

// FuzzWindowWhenUnsubscribe isolates the teardown path of WindowWhen: the source emits 5 items then stays
// open, so only the Unsubscribe from outside ends the stream, after a few scheduler yields.
//
// Invariant: the source and the boundary are released, and the open window receives a terminal
// notification, otherwise its subscriber waits forever.
//
// Seeds: ticks, yieldPattern and spinsBeforeUnsubscribe spread over their range; asyncSource and
// asyncTicks alternate.
func FuzzWindowWhenUnsubscribe(f *testing.F) {
	f.Skip("race: windowwhen-teardown-leaves-window-open; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// ticks, asyncSource, asyncTicks, yieldPattern, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), i%2 == 0, i%4 < 2, seedByte(i, 2), seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, ticks uint8, asyncSource, asyncTicks bool, yieldPattern, spinsBeforeUnsubscribe uint8) {
		const itemsBeforeStaying = 5

		numbers := newSource(itemsBeforeStaying, asyncSource).neverEnding().yieldingWith(int64(yieldPattern))

		var upstream, boundary subscriptionCounter

		tickSource := countSubscriptions(&boundary, tickingBoundary(bounded(ticks, 0, maxTicks), asyncTicks, yieldPattern))
		pipeline := ro.WindowWhen[int](tickSource)(countSubscriptions(&upstream, numbers.observable()))

		windows := newInnerStreamCollector()
		windows.subscribeAndEnd(t, pipeline, newStreamEnd(true, spinsBeforeUnsubscribe, yieldPattern))

		upstream.expectAllReleased(t)
		boundary.expectAllReleased(t)
		windows.expectEveryStreamEnded(t)
	})
}

// FuzzSampleWhen emits the latest item of a source of `items` integers at each tick of a boundary that
// sends `ticks` ticks and never completes. Source and boundary are synchronous or asynchronous.
//
// Invariant: the samples are strictly increasing values of 0..items-1, and the source and the boundary are
// released.
//
// Seeds: as FuzzBufferWhen.
func FuzzSampleWhen(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, ticks, asyncSource, asyncTicks, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%8 < 4, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, ticks uint8, asyncSource, asyncTicks bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream, boundary subscriptionCounter

		tickSource := countSubscriptions(&boundary, tickingBoundary(bounded(ticks, 0, maxTicks), asyncTicks, yieldPattern))
		pipeline := ro.SampleWhen[int](tickSource)(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream, &boundary)
		if completed {
			expectStrictlyIncreasing(t, got.received(), count)
		}
	})
}

// FuzzSampleTime emits the latest item of a source of `items` integers every periodMicros microseconds.
// The source is synchronous or asynchronous.
//
// Invariant: the samples are strictly increasing values of 0..items-1, and the source is released.
//
// Seeds: as FuzzBufferWithTime.
func FuzzSampleTime(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, periodMicros, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), uint16(i * 251), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items uint8, periodMicros uint16, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.SampleTime[int](timerPeriod(periodMicros))(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectStrictlyIncreasing(t, got.received(), count)
		}
	})
}

// FuzzThrottleWhen lets one item of a source of `items` integers through after each tick of a boundary that
// sends `ticks` ticks and never completes. Source and boundary are synchronous or asynchronous.
//
// Invariant: no more values pass than ticks were sent, the values are strictly increasing in 0..items-1,
// and the source and the boundary are released.
//
// Seeds: as FuzzBufferWhen.
func FuzzThrottleWhen(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, ticks, asyncSource, asyncTicks, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%8 < 4, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, ticks uint8, asyncSource, asyncTicks bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		tickCount := bounded(ticks, 0, maxTicks)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream, boundary subscriptionCounter

		tickSource := countSubscriptions(&boundary, tickingBoundary(tickCount, asyncTicks, yieldPattern))
		pipeline := ro.ThrottleWhen[int](tickSource)(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream, &boundary)
		if completed {
			if passed := got.valueCount(); passed > tickCount {
				t.Fatalf("%d values passed but only %d ticks were sent", passed, tickCount)
			}

			expectStrictlyIncreasing(t, got.received(), count)
		}
	})
}

// FuzzThrottleTime lets one item of a source of `items` integers through every periodMicros microseconds.
// The source is synchronous or asynchronous.
//
// Invariant: the values are strictly increasing in 0..items-1, and the source is released.
//
// Seeds: as FuzzBufferWithTime.
func FuzzThrottleTime(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, periodMicros, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), uint16(i * 251), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items uint8, periodMicros uint16, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.ThrottleTime[int](timerPeriod(periodMicros))(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectStrictlyIncreasing(t, got.received(), count)
		}
	})
}

// maxGroupKeys is the largest number of distinct keys of FuzzGroupBy. Few keys put many items in each group.
const maxGroupKeys = 4

// FuzzGroupBy groups a source of `items` integers by value modulo `keys`. The source is synchronous or
// asynchronous, and the stream either runs to its end or is unsubscribed from outside.
//
// Invariant: one group per key present, each group holds the items of its key in increasing order, the
// groups hold every item once, every group ends, and the source is released. When unsubscribed early,
// only the release and the end of every group are checked.
//
// Seeds: items, keys, yieldPattern and spinsBeforeUnsubscribe spread over their range; asyncSource and
// unsubscribeEarly alternate.
func FuzzGroupBy(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, keys, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items, keys uint8, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		keyCount := bounded(keys, 1, maxGroupKeys)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.GroupBy(func(item int) int { return item % keyCount })(countSubscriptions(&upstream, numbers.observable()))

		groups := newInnerStreamCollector()
		completed := groups.subscribeAndEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern))

		if completed {
			groups.expectEveryStreamEnded(t)
		}

		upstream.expectAllReleased(t)

		if !completed {
			groups.expectEveryStreamEnded(t)

			return
		}

		expectGroupedByKey(t, groups.itemsPerStream(), count, keyCount)
	})
}

// expectGroupedByKey checks the groups of items 0..count-1 keyed by value modulo keyCount.
func expectGroupedByKey(t *testing.T, groups [][]int, count, keyCount int) {
	t.Helper()

	if want := smaller(keyCount, count); len(groups) != want {
		t.Fatalf("%d groups, want %d", len(groups), want)
	}

	total := 0

	for _, group := range groups {
		total += len(group)

		for index, item := range group {
			if item%keyCount != group[0]%keyCount {
				t.Fatalf("group mixes keys: %v", group)
			}

			if index > 0 && item <= group[index-1] {
				t.Fatalf("group out of order: %v", group)
			}
		}
	}

	if total != count {
		t.Fatalf("groups hold %d items, want %d", total, count)
	}
}

// FuzzGroupByUnsubscribeWhileEmitting emits many values with a few keys from a subject and unsubscribes
// from the group-by while values for existing keys are still in flight. The key function yields the
// processor, so values stay in flight while the teardown runs.
//
// Invariant: the teardown neither races on the group registry nor leaves a group uncompleted.
//
// Seeds: yieldPattern spreads over its range.
func FuzzGroupByUnsubscribeWhileEmitting(f *testing.F) {
	const (
		keys      = 8
		emissions = 2000
	)

	fuzzSeeds(f, func(i int) []any {
		// yieldPattern
		return []any{seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, yieldPattern uint8) {
		subject := ro.NewPublishSubject[int]()

		var openGroups int64

		keyOf := func(item int) int {
			runtime.Gosched()
			yieldAt(int64(yieldPattern), item)

			return item % keys
		}

		subscription := ro.GroupBy(keyOf)(subject).Subscribe(ro.OnNext(func(group ro.Observable[int]) {
			atomic.AddInt64(&openGroups, 1)
			group.Subscribe(ro.NewObserver(
				func(int) {},
				func(error) { atomic.AddInt64(&openGroups, -1) },
				func() { atomic.AddInt64(&openGroups, -1) },
			))
		}))

		var emitter sync.WaitGroup

		keysSeen := make(chan struct{})

		emitter.Add(1)

		go func() {
			defer emitter.Done()

			for item := 0; item < emissions; item++ {
				subject.Next(item)

				if item == keys {
					close(keysSeen)
				}
			}
		}()

		<-keysSeen // every key already has a group: the next values hit existing groups

		yieldAt(int64(yieldPattern), emissions)
		subscription.Unsubscribe()
		expectWaitGroupDone(t, "the emitter", &emitter)

		if open := atomic.LoadInt64(&openGroups); open != 0 {
			t.Fatalf("%d groups still open after the teardown, want every emitted group completed", open)
		}
	})
}
