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
	"reflect"
	"testing"

	"github.com/samber/lo"
	"github.com/samber/ro"
)

// FuzzMergeAll merges the inner sources selected by an outer source, through an outer Observable of
// Observables. Inners and outer are synchronous or asynchronous, and the stream runs to its end.
//
// Invariant: every inner value arrives exactly once, the stream completes once, and every inner and
// outer subscription is released.
//
// Seeds: outerItems and firstInnerSize spread over their range; asyncOuter, asyncEvenInners and
// asyncOddInners alternate.
func FuzzMergeAll(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runToCompletion(t, ro.MergeAll[int]()(ro.Map(sources.project)(sources.outerItems())), sources.upstreams()...)

		sources.expectEveryValue(t, got, false)
	})
}

// FuzzMergeAllTake is FuzzMergeAll closed from inside the pipeline by Take(takeLimit).
//
// Invariant: Take delivers min(takeLimit, available) values, and every inner and outer subscription is
// released.
//
// Seeds: as FuzzMergeAll, with takeLimit spread over its range.
func FuzzMergeAllTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, takeLimit uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, ro.MergeAll[int]()(ro.Map(sources.project)(sources.outerItems())), limit, sources.upstreams()...)

		sources.expectTakenValues(t, got, limit, false)
	})
}

// FuzzMergeAllUnsubscribe is FuzzMergeAll unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: no value is invented, and every inner and outer subscription is released.
//
// Seeds: as FuzzMergeAll, with unsubscribeAfter spread over its range.
func FuzzMergeAllUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, unsubscribeAfter uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runThenUnsubscribe(t, ro.MergeAll[int]()(ro.Map(sources.project)(sources.outerItems())), bounded(unsubscribeAfter, 1, maxItemsBeforeStop), sources.upstreams()...)

		sources.expectOnlyEmittedValues(t, got, false)
	})
}

// FuzzConcatAll concatenates the inner sources selected by an outer source, through an outer Observable of
// Observables. Inners and outer are synchronous or asynchronous, and the stream runs to its end.
//
// Invariant: every inner value arrives exactly once, inner after inner in order, the stream completes
// once, and every inner and outer subscription is released.
//
// Seeds: as FuzzMergeAll.
func FuzzConcatAll(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runToCompletion(t, ro.ConcatAll[int]()(ro.Map(sources.project)(sources.outerItems())), sources.upstreams()...)

		sources.expectEveryValue(t, got, true)
	})
}

// FuzzConcatAllTake is FuzzConcatAll closed from inside the pipeline by Take(takeLimit).
//
// Invariant: Take delivers the first min(takeLimit, available) values in order, and every inner and
// outer subscription is released.
//
// Seeds: as FuzzMergeAllTake.
func FuzzConcatAllTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, takeLimit uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, ro.ConcatAll[int]()(ro.Map(sources.project)(sources.outerItems())), limit, sources.upstreams()...)

		sources.expectTakenValues(t, got, limit, true)
	})
}

// FuzzConcatAllUnsubscribe is FuzzConcatAll unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: the values received are the first values in order, and every inner and outer subscription
// is released.
//
// Seeds: as FuzzMergeAllUnsubscribe.
func FuzzConcatAllUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// outerItems, firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, outerItems, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool, unsubscribeAfter uint8) {
		sources := newHigherOrderSources(bounded(outerItems, 1, maxItems), firstInnerSize, asyncOuter, asyncEvenInners, asyncOddInners)

		got := runThenUnsubscribe(t, ro.ConcatAll[int]()(ro.Map(sources.project)(sources.outerItems())), bounded(unsubscribeAfter, 1, maxItemsBeforeStop), sources.upstreams()...)

		sources.expectOnlyEmittedValues(t, got, true)
	})
}

// FuzzConcatWith concatenates `sources` sources given as arguments. Even and odd sources are synchronous
// or asynchronous independently, and the stream runs to its end.
//
// Invariant: every value arrives exactly once, source after source in order, the stream completes once,
// and every source subscription is released.
//
// Seeds: sources and firstInnerSize spread over their range; asyncEvenInners and asyncOddInners alternate.
func FuzzConcatWith(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sources, firstInnerSize, asyncEvenInners, asyncOddInners
		return []any{seedByte(i, 0), seedByte(i, 1), i%3 == 0, i%5 < 2}
	})

	f.Fuzz(func(t *testing.T, sourceCount, firstInnerSize uint8, asyncEvenInners, asyncOddInners bool) {
		sources := newHigherOrderSources(bounded(sourceCount, 1, maxConcatSources), firstInnerSize, false, asyncEvenInners, asyncOddInners)

		got := runToCompletion(t, ro.ConcatWith(sources.inners[1:]...)(sources.inners[0]), sources.upstreams()...)

		sources.expectEveryValue(t, got, true)
	})
}

// FuzzConcatWithTake is FuzzConcatWith closed from inside the pipeline by Take(takeLimit).
//
// Invariant: Take delivers the first min(takeLimit, available) values in order, and every source
// subscription is released.
//
// Seeds: as FuzzConcatWith, with takeLimit spread over its range.
func FuzzConcatWithTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sources, firstInnerSize, asyncEvenInners, asyncOddInners, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, sourceCount, firstInnerSize uint8, asyncEvenInners, asyncOddInners bool, takeLimit uint8) {
		sources := newHigherOrderSources(bounded(sourceCount, 1, maxConcatSources), firstInnerSize, false, asyncEvenInners, asyncOddInners)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, ro.ConcatWith(sources.inners[1:]...)(sources.inners[0]), limit, sources.upstreams()...)

		sources.expectTakenValues(t, got, limit, true)
	})
}

// FuzzConcatWithUnsubscribe is FuzzConcatWith unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: the values received are the first values in order, and every source subscription is released.
//
// Seeds: as FuzzConcatWith, with unsubscribeAfter spread over its range.
func FuzzConcatWithUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sources, firstInnerSize, asyncEvenInners, asyncOddInners, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), i%3 == 0, i%5 < 2, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, sourceCount, firstInnerSize uint8, asyncEvenInners, asyncOddInners bool, unsubscribeAfter uint8) {
		sources := newHigherOrderSources(bounded(sourceCount, 1, maxConcatSources), firstInnerSize, false, asyncEvenInners, asyncOddInners)

		got := runThenUnsubscribe(t, ro.ConcatWith(sources.inners[1:]...)(sources.inners[0]), bounded(unsubscribeAfter, 1, maxItemsBeforeStop), sources.upstreams()...)

		sources.expectOnlyEmittedValues(t, got, true)
	})
}

const (
	// maxPrefixes is the largest number of values StartWith prepends.
	maxPrefixes = 3

	// maxMergedItems is the largest size of each of the two merged sources.
	maxMergedItems = 4

	// firstMergedValueBase and secondMergedValueBase give the two merged sources disjoint values.
	firstMergedValueBase  = 1000
	secondMergedValueBase = 2000
)

// startWithScenario is StartWith over the merge of two sources. The prefixes are negative so they never
// collide with a source value: prefix i is -1-i.
type startWithScenario struct {
	firstCounter  subscriptionCounter
	secondCounter subscriptionCounter

	prefixes []int
	pipeline ro.Observable[int]

	// total is the number of values the pipeline emits when it runs to its end.
	total int
}

func newStartWithScenario(prefixCount, firstItems, secondItems uint8, asyncFirst, asyncSecond bool) *startWithScenario {
	scenario := &startWithScenario{}

	for index := 0; index < bounded(prefixCount, 0, maxPrefixes); index++ {
		scenario.prefixes = append(scenario.prefixes, -1-index)
	}

	firstSize := bounded(firstItems, 0, maxMergedItems)
	secondSize := bounded(secondItems, 0, maxMergedItems)
	scenario.total = len(scenario.prefixes) + firstSize + secondSize

	first := countSubscriptions(&scenario.firstCounter, newSource(firstSize, asyncFirst).startingAt(firstMergedValueBase).observable())
	second := countSubscriptions(&scenario.secondCounter, newSource(secondSize, asyncSecond).startingAt(secondMergedValueBase).observable())
	scenario.pipeline = ro.StartWith(scenario.prefixes...)(ro.Merge(first, second))

	return scenario
}

func (s *startWithScenario) upstreams() []*subscriptionCounter {
	return []*subscriptionCounter{&s.firstCounter, &s.secondCounter}
}

// expectPrefixesFirst checks that the values received start with the prefixes, in order.
func (s *startWithScenario) expectPrefixesFirst(t *testing.T, got *recorder[int]) {
	t.Helper()

	values := got.received()
	for index := 0; index < smaller(len(s.prefixes), len(values)); index++ {
		if values[index] != s.prefixes[index] {
			t.Fatalf("prefix %d is %d, want %d", index, values[index], s.prefixes[index])
		}
	}
}

// FuzzStartWith prepends `prefixCount` values to the merge of two sources. The sources are synchronous
// or asynchronous, and the stream runs to its end.
//
// Invariant: the prefixes come first and in order, every value arrives exactly once, the stream
// completes once, and both sources are released.
//
// Seeds: prefixCount, firstItems and secondItems spread over their range; asyncFirst and asyncSecond
// alternate.
func FuzzStartWith(f *testing.F) {
	f.Skip("race: startwith-unsafe-merge; remove when fixed") // Fails on main: overlapping notifications on the downstream observer: 1

	fuzzSeeds(f, func(i int) []any {
		// prefixCount, firstItems, secondItems, asyncFirst, asyncSecond
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, prefixCount, firstItems, secondItems uint8, asyncFirst, asyncSecond bool) {
		scenario := newStartWithScenario(prefixCount, firstItems, secondItems, asyncFirst, asyncSecond)

		got := runToCompletion(t, scenario.pipeline, scenario.upstreams()...)

		scenario.expectPrefixesFirst(t, got)

		if received := got.valueCount(); received != scenario.total {
			t.Fatalf("got %d/%d values", received, scenario.total)
		}

		got.expectCompletedOnce(t)
	})
}

// FuzzStartWithTake is FuzzStartWith closed from inside the pipeline by Take(takeLimit).
//
// Invariant: the prefixes come first and in order, Take delivers min(takeLimit, total) values, and both
// sources are released.
//
// Seeds: as FuzzStartWith, with takeLimit spread over its range.
func FuzzStartWithTake(f *testing.F) {
	f.Skip("race: startwith-unsafe-merge; remove when fixed") // Fails on main: overlapping notifications on the downstream observer: 1

	fuzzSeeds(f, func(i int) []any {
		// prefixCount, firstItems, secondItems, asyncFirst, asyncSecond, takeLimit
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%3 == 0, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, prefixCount, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, takeLimit uint8) {
		scenario := newStartWithScenario(prefixCount, firstItems, secondItems, asyncFirst, asyncSecond)
		limit := bounded(takeLimit, 1, maxItemsBeforeStop)

		got := runWithTake(t, scenario.pipeline, limit, scenario.upstreams()...)

		scenario.expectPrefixesFirst(t, got)

		if received, want := got.valueCount(), smaller(limit, scenario.total); received != want {
			t.Fatalf("take(%d) delivered %d of %d", limit, received, scenario.total)
		}
	})
}

// FuzzStartWithUnsubscribe is FuzzStartWith unsubscribed from outside after unsubscribeAfter values.
//
// Invariant: the prefixes come first and in order, and both sources are released.
//
// Seeds: as FuzzStartWith, with unsubscribeAfter spread over its range.
func FuzzStartWithUnsubscribe(f *testing.F) {
	f.Skip("race: startwith-unsafe-merge; remove when fixed") // Fails on main: overlapping notifications on the downstream observer: 1

	fuzzSeeds(f, func(i int) []any {
		// prefixCount, firstItems, secondItems, asyncFirst, asyncSecond, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%3 == 0, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, prefixCount, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, unsubscribeAfter uint8) {
		scenario := newStartWithScenario(prefixCount, firstItems, secondItems, asyncFirst, asyncSecond)

		got := runThenUnsubscribe(t, scenario.pipeline, bounded(unsubscribeAfter, 1, maxItemsBeforeStop), scenario.upstreams()...)

		scenario.expectPrefixesFirst(t, got)
	})
}

// A race has at least 2 sources: one winner, one loser.
const (
	minRaceSources = 2
	maxRaceSources = 4
)

// raceSources are the contestants of a race. Source i emits i*innerValueStride+0, +1, ... so a value tells
// which source emitted it. Source i has 1 + (firstSize+i) % 3 items, at least one so that a loser
// of an endless race still has something to race with. Even and odd sources are synchronous or
// asynchronous independently. An endless source never completes after its items.
type raceSources struct {
	counters []*subscriptionCounter
	sources  []ro.Observable[int]
}

func newRaceSources(count int, firstSize uint8, asyncEven, asyncOdd, endless bool) *raceSources {
	const itemCycle = 3

	contestants := &raceSources{}

	for index := 0; index < count; index++ {
		async := asyncOdd
		if index%2 == 0 {
			async = asyncEven
		}

		contestant := newSource(1+(int(firstSize)+index)%itemCycle, async).startingAt(index * innerValueStride).yieldingWith(int64(index))
		if endless {
			contestant.neverEnding()
		}

		counter := &subscriptionCounter{}
		contestants.counters = append(contestants.counters, counter)
		contestants.sources = append(contestants.sources, countSubscriptions(counter, contestant.observable()))
	}

	return contestants
}

// expectOneWinner checks that every value comes from the same source.
func expectOneWinner(t *testing.T, got *recorder[int]) {
	t.Helper()

	values := got.received()
	for _, value := range values {
		if value/innerValueStride != values[0]/innerValueStride {
			t.Fatalf("values of two sources mixed: %v", values)
		}
	}
}

// runFinishingRace runs a race whose sources complete, to its end.
func (c *raceSources) runFinishingRace(t *testing.T, race ro.Observable[int]) {
	t.Helper()

	got := runToCompletion(t, race, c.counters...)

	expectOneWinner(t, got)
}

// runEndlessRace runs a race whose sources never complete: once the winner emitted, every loser must be
// unsubscribed, although the winner itself stays subscribed until the downstream unsubscribes.
func (c *raceSources) runEndlessRace(t *testing.T, race ro.Observable[int]) {
	t.Helper()

	got := newRecorder[int]()
	subscription := subscribeInBackground(context.Background(), race, got)

	waitUntil(t, "the winner's first item", func() bool { return got.valueCount() >= 1 })

	winner := got.received()[0] / innerValueStride

	for index, counter := range c.counters {
		if index != winner {
			counter := counter
			waitUntil(t, "a loser to be unsubscribed", func() bool { return counter.activeCount() == 0 })
		}
	}

	subscription.unsubscribeAfterItems(t, got, 1)

	expectReleasedWithContract(t, got, c.counters)
	expectOneWinner(t, got)
}

// FuzzRace races `sources` sources that complete. Sources are synchronous or asynchronous.
//
// Invariant: all values come from one source, the winner, the stream completes once, and every source
// subscription is released.
//
// Seeds: sources and firstSize spread over their range; asyncEvenSources and asyncOddSources alternate.
func FuzzRace(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sources, firstSize, asyncEvenSources, asyncOddSources
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, sources, firstSize uint8, asyncEvenSources, asyncOddSources bool) {
		contestants := newRaceSources(bounded(sources, minRaceSources, maxRaceSources), firstSize, asyncEvenSources, asyncOddSources, false)

		contestants.runFinishingRace(t, ro.Race(contestants.sources...))
	})
}

// FuzzRaceWith is FuzzRace through RaceWith: the first source races the others.
//
// Invariant: as FuzzRace.
//
// Seeds: as FuzzRace.
func FuzzRaceWith(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sources, firstSize, asyncEvenSources, asyncOddSources
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, sources, firstSize uint8, asyncEvenSources, asyncOddSources bool) {
		contestants := newRaceSources(bounded(sources, minRaceSources, maxRaceSources), firstSize, asyncEvenSources, asyncOddSources, false)

		contestants.runFinishingRace(t, ro.RaceWith(contestants.sources[1:]...)(contestants.sources[0]))
	})
}

// FuzzRaceLosersUnsubscribed races `sources` sources that never complete after their items.
//
// Invariant: once the winner emitted, every loser is unsubscribed; all values come from the winner; every
// source subscription is released when the downstream unsubscribes.
//
// Seeds: as FuzzRace.
func FuzzRaceLosersUnsubscribed(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // Fails on main: timeout waiting for: upstream 0 to drop to 0 active subscriptions

	fuzzSeeds(f, func(i int) []any {
		// sources, firstSize, asyncEvenSources, asyncOddSources
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, sources, firstSize uint8, asyncEvenSources, asyncOddSources bool) {
		contestants := newRaceSources(bounded(sources, minRaceSources, maxRaceSources), firstSize, asyncEvenSources, asyncOddSources, true)

		contestants.runEndlessRace(t, ro.Race(contestants.sources...))
	})
}

// FuzzRaceWithLosersUnsubscribed is FuzzRaceLosersUnsubscribed through RaceWith.
//
// Invariant: as FuzzRaceLosersUnsubscribed.
//
// Seeds: as FuzzRace.
func FuzzRaceWithLosersUnsubscribed(f *testing.F) {
	f.Skip("race: race-winner-not-retained; remove when fixed") // Fails on main: timeout waiting for: upstream 0 to drop to 0 active subscriptions

	fuzzSeeds(f, func(i int) []any {
		// sources, firstSize, asyncEvenSources, asyncOddSources
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, sources, firstSize uint8, asyncEvenSources, asyncOddSources bool) {
		contestants := newRaceSources(bounded(sources, minRaceSources, maxRaceSources), firstSize, asyncEvenSources, asyncOddSources, true)

		contestants.runEndlessRace(t, ro.RaceWith(contestants.sources[1:]...)(contestants.sources[0]))
	})
}

// FuzzPairwise pairs each item of a source of `items` integers with the previous one. The source is
// synchronous or asynchronous, and the stream either runs to its end or is unsubscribed from outside.
//
// Invariant: the pairs are exactly [0 1], [1 2], ..., and the source is released. When unsubscribed
// early, only the release and the observer contract are checked.
//
// Seeds: items, yieldPattern and spinsBeforeUnsubscribe spread over their range; asyncSource and
// unsubscribeEarly alternate.
func FuzzPairwise(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, asyncSource, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), i%2 == 0, seedByte(i, 2), i%4 >= 2, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, items uint8, asyncSource bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		count := bounded(items, 0, maxItems)
		numbers := newSource(count, asyncSource).yieldingWith(int64(yieldPattern))

		var upstream subscriptionCounter

		pipeline := ro.Pairwise[int]()(countSubscriptions(&upstream, numbers.observable()))

		got, completed := observeUntilEnd(t, pipeline, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &upstream)
		if completed {
			expectConsecutivePairs(t, got.received(), count)
		}
	})
}

// expectConsecutivePairs checks the pairs of items 0..count-1: [0 1], [1 2], ... An empty or one-item source has none.
func expectConsecutivePairs(t *testing.T, pairs [][]int, count int) {
	t.Helper()

	want := count - 1
	if want < 0 {
		want = 0
	}

	if len(pairs) != want {
		t.Fatalf("%d pairs, want %d", len(pairs), want)
	}

	for index, pair := range pairs {
		if len(pair) != 2 || pair[0] != index || pair[1] != index+1 {
			t.Fatalf("pair %d is %v, want [%d %d]", index, pair, index, index+1)
		}
	}
}

// FuzzZip2 zips two sources of firstItems and secondItems integers. Each source is synchronous or
// asynchronous, and the stream either runs to its end or is unsubscribed from outside.
//
// Invariant: the pairs are (0,0), (1,1), ..., one per item of the shorter source, in order, and both
// sources are released. When unsubscribed early, only the release and the observer contract are checked.
//
// Seeds: firstItems, secondItems, yieldPattern and spinsBeforeUnsubscribe spread over their range;
// asyncFirst, asyncSecond and unsubscribeEarly alternate.
func FuzzZip2(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, asyncFirst, asyncSecond, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 2), i%8 >= 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		firstCount := bounded(firstItems, 0, maxItems)
		secondCount := bounded(secondItems, 0, maxItems)

		var firstUpstream, secondUpstream subscriptionCounter

		first := countSubscriptions(&firstUpstream, newSource(firstCount, asyncFirst).yieldingWith(int64(yieldPattern)).observable())
		second := countSubscriptions(&secondUpstream, newSource(secondCount, asyncSecond).yieldingWith(int64(yieldPattern)+1).observable())

		got, completed := observeUntilEnd(t, ro.Zip2(first, second), newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), &firstUpstream, &secondUpstream)
		if !completed {
			return
		}

		pairs := got.received()
		if want := smaller(firstCount, secondCount); len(pairs) != want {
			t.Fatalf("%d pairs, want %d: %v", len(pairs), want, pairs)
		}

		for index, pair := range pairs {
			if pair.A != index || pair.B != index {
				t.Fatalf("pair %d is %v, want (%d,%d): out of order in %v", index, pair, index, index, pairs)
			}
		}
	})
}

// FuzzZipVariadic zips three sources of firstItems, secondItems and thirdItems integers through the
// variadic Zip. Each source is synchronous or asynchronous, and the stream either runs to its end or is
// unsubscribed from outside.
//
// Invariant: the tuples are [0 0 0], [1 1 1], ..., one per item of the shortest source, in order, and every
// source is released. When unsubscribed early, only the release and the observer contract are checked.
//
// Seeds: the item counts, yieldPattern and spinsBeforeUnsubscribe spread over their range; the async
// flags and unsubscribeEarly alternate.
func FuzzZipVariadic(f *testing.F) {
	f.Skip("race: zip-pairs-delivered-out-of-order; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, thirdItems, asyncFirst, asyncSecond, asyncThird, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, i%8 < 4, seedByte(i, 3), i%16 >= 8, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems, thirdItems uint8, asyncFirst, asyncSecond, asyncThird bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		counts := []int{bounded(firstItems, 0, maxItems), bounded(secondItems, 0, maxItems), bounded(thirdItems, 0, maxItems)}
		asyncSources := []bool{asyncFirst, asyncSecond, asyncThird}
		upstreams := make([]*subscriptionCounter, len(counts))
		sources := make([]ro.Observable[int], len(counts))

		for index, count := range counts {
			upstreams[index] = &subscriptionCounter{}
			sources[index] = countSubscriptions(upstreams[index], newSource(count, asyncSources[index]).yieldingWith(int64(yieldPattern)+int64(index)).observable())
		}

		got, completed := observeUntilEnd(t, ro.Zip(sources...), newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), upstreams...)
		if !completed {
			return
		}

		tuples := got.received()
		if want := smaller(counts[0], smaller(counts[1], counts[2])); len(tuples) != want {
			t.Fatalf("%d tuples, want %d: %v", len(tuples), want, tuples)
		}

		for index, tuple := range tuples {
			if len(tuple) != len(counts) || tuple[0] != index || tuple[1] != index || tuple[2] != index {
				t.Fatalf("tuple %d is %v, want all %d: out of order in %v", index, tuple, index, tuples)
			}
		}
	})
}

// latestPairs is what CombineLatest2 emitted for two sources of firstCount and secondCount items.
type latestPairs struct {
	tuples      []lo.Tuple2[int, int]
	firstCount  int
	secondCount int
}

// runCombineLatest2 combines two sources of firstItems and secondItems integers, then ends the stream as
// asked. It checks what every CombineLatest2 target shares: nothing is emitted while a source is empty,
// and something is emitted once both sources emitted. It reports whether the tuples are worth checking
// further, which is when the stream ran to its end and both sources emitted.
func runCombineLatest2(t *testing.T, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, end streamEnd) (pairs latestPairs, worthChecking bool) {
	t.Helper()

	pairs.firstCount = bounded(firstItems, 0, maxItems)
	pairs.secondCount = bounded(secondItems, 0, maxItems)

	var firstUpstream, secondUpstream subscriptionCounter

	first := countSubscriptions(&firstUpstream, newSource(pairs.firstCount, asyncFirst).yieldingWith(int64(end.yieldPattern)).observable())
	second := countSubscriptions(&secondUpstream, newSource(pairs.secondCount, asyncSecond).yieldingWith(int64(end.yieldPattern)+1).observable())

	got, completed := observeUntilEnd(t, ro.CombineLatest2(first, second), end, &firstUpstream, &secondUpstream)
	if !completed {
		return pairs, false
	}

	pairs.tuples = got.received()

	if pairs.firstCount == 0 || pairs.secondCount == 0 {
		if len(pairs.tuples) != 0 {
			t.Fatalf("emitted %v although a source is empty", pairs.tuples)
		}

		return pairs, false
	}

	if len(pairs.tuples) == 0 {
		t.Fatal("no tuple emitted although both sources emitted")
	}

	return pairs, true
}

// FuzzCombineLatest2 combines two sources of firstItems and secondItems integers. Each source is
// synchronous or asynchronous, and the stream either runs to its end or is unsubscribed from outside.
//
// Invariant: the last tuple holds the latest item of both sources, and both sources are released.
//
// Seeds: firstItems, secondItems, yieldPattern and spinsBeforeUnsubscribe spread over their range;
// asyncFirst, asyncSecond and unsubscribeEarly alternate.
func FuzzCombineLatest2(f *testing.F) {
	f.Skip("race: combinelatest-stale-last-tuple; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, asyncFirst, asyncSecond, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 2), i%8 >= 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		pairs, worthChecking := runCombineLatest2(t, firstItems, secondItems, asyncFirst, asyncSecond, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern))
		if !worthChecking {
			return
		}

		if last := pairs.tuples[len(pairs.tuples)-1]; last.A != pairs.firstCount-1 || last.B != pairs.secondCount-1 {
			t.Fatalf("stale last tuple %v, want (%d,%d)", last, pairs.firstCount-1, pairs.secondCount-1)
		}
	})
}

// FuzzCombineLatest2Order is FuzzCombineLatest2 checking the order of the tuples.
//
// Invariant: a tuple never goes back to an older item of either source, and both sources are released.
//
// Seeds: as FuzzCombineLatest2.
func FuzzCombineLatest2Order(f *testing.F) {
	f.Skip("race: combinelatest-out-of-order-tuples; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, asyncFirst, asyncSecond, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 2), i%8 >= 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		pairs, worthChecking := runCombineLatest2(t, firstItems, secondItems, asyncFirst, asyncSecond, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern))
		if !worthChecking {
			return
		}

		for index := 1; index < len(pairs.tuples); index++ {
			if pairs.tuples[index].A < pairs.tuples[index-1].A || pairs.tuples[index].B < pairs.tuples[index-1].B {
				t.Fatalf("tuple went backwards: %v then %v", pairs.tuples[index-1], pairs.tuples[index])
			}
		}
	})
}

// FuzzCombineLatest2Duplicate is FuzzCombineLatest2 checking for repeated tuples.
//
// Invariant: no two consecutive tuples are identical, and both sources are released.
//
// Seeds: as FuzzCombineLatest2.
func FuzzCombineLatest2Duplicate(f *testing.F) {
	f.Skip("race: combinelatest-duplicate-tuples; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, asyncFirst, asyncSecond, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, seedByte(i, 2), i%8 >= 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems uint8, asyncFirst, asyncSecond bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		pairs, worthChecking := runCombineLatest2(t, firstItems, secondItems, asyncFirst, asyncSecond, newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern))
		if !worthChecking {
			return
		}

		for index := 1; index < len(pairs.tuples); index++ {
			if pairs.tuples[index] == pairs.tuples[index-1] {
				t.Fatalf("duplicate tuple %v", pairs.tuples[index])
			}
		}
	})
}

// FuzzCombineLatestAll combines three sources of 1..maxItems integers handed over as an Observable of
// Observables. Each source is synchronous or asynchronous, and the stream either runs to its end or is
// unsubscribed from outside.
//
// Invariant: the last tuple holds the latest item of every source, and every source is released. When
// unsubscribed early, only the release and the observer contract are checked.
//
// Seeds: the item counts, yieldPattern and spinsBeforeUnsubscribe spread over their range; the async
// flags and unsubscribeEarly alternate.
func FuzzCombineLatestAll(f *testing.F) {
	f.Skip("race: combinelatestall-stale-last-tuple; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// firstItems, secondItems, thirdItems, asyncFirst, asyncSecond, asyncThird, yieldPattern, unsubscribeEarly, spinsBeforeUnsubscribe
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, i%8 < 4, seedByte(i, 3), i%16 >= 8, seedByte(i, 4)}
	})

	f.Fuzz(func(t *testing.T, firstItems, secondItems, thirdItems uint8, asyncFirst, asyncSecond, asyncThird bool, yieldPattern uint8, unsubscribeEarly bool, spinsBeforeUnsubscribe uint8) {
		counts := []int{bounded(firstItems, 1, maxItems), bounded(secondItems, 1, maxItems), bounded(thirdItems, 1, maxItems)}
		asyncSources := []bool{asyncFirst, asyncSecond, asyncThird}
		upstreams := make([]*subscriptionCounter, len(counts))
		sources := make([]ro.Observable[int], len(counts))

		for index, count := range counts {
			upstreams[index] = &subscriptionCounter{}
			sources[index] = countSubscriptions(upstreams[index], newSource(count, asyncSources[index]).yieldingWith(int64(yieldPattern)+int64(index)).observable())
		}

		got, completed := observeUntilEnd(t, ro.CombineLatestAll[int]()(ro.Just(sources...)), newStreamEnd(unsubscribeEarly, spinsBeforeUnsubscribe, yieldPattern), upstreams...)
		if !completed {
			return
		}

		tuples := got.received()
		if len(tuples) == 0 {
			t.Fatal("no tuple emitted although every source emitted")
		}

		last := tuples[len(tuples)-1]
		for index, count := range counts {
			if len(last) != len(counts) || last[index] != count-1 {
				t.Fatalf("stale last tuple %v, want latest values of every source", last)
			}
		}
	})
}

// zipVariant wraps one zip operator behind a common shape, so a target runs against all of them.
type zipVariant struct {
	name  string
	arity int
	// zip emits each group of paired values as a slice, whatever the operator's output type.
	zip func(sources []ro.Observable[int]) ro.Observable[[]int]
}

func zipVariants() []zipVariant {
	return []zipVariant{
		{"Zip", 2, func(s []ro.Observable[int]) ro.Observable[[]int] { return ro.Zip(s...) }},
		{"ZipAll", 2, func(s []ro.Observable[int]) ro.Observable[[]int] { return ro.ZipAll[int]()(ro.Just(s...)) }},
		{"Zip2", 2, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple2[int, int]) []int { return []int{v.A, v.B} })(ro.Zip2(s[0], s[1]))
		}},
		{"Zip3", 3, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple3[int, int, int]) []int { return []int{v.A, v.B, v.C} })(ro.Zip3(s[0], s[1], s[2]))
		}},
		{"Zip4", 4, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple4[int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D} })(ro.Zip4(s[0], s[1], s[2], s[3]))
		}},
		{"Zip5", 5, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple5[int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E} })(ro.Zip5(s[0], s[1], s[2], s[3], s[4]))
		}},
		{"Zip6", 6, func(s []ro.Observable[int]) ro.Observable[[]int] {
			return ro.Map(func(v lo.Tuple6[int, int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E, v.F} })(ro.Zip6(s[0], s[1], s[2], s[3], s[4], s[5]))
		}},
	}
}

// FuzzZipFutureCompletion zips Futures that all resolve at about the same moment, for every zip variant
// (Zip, ZipAll, Zip2 to Zip6). A source completing while another goroutine delivers the last group must
// not drop it. The window is a few instructions wide, so every seed is one more attempt to hit it.
//
// Invariant: each variant emits exactly one group, [0 1 ... arity-1], and no error.
//
// Seeds: yieldPattern spreads over its range.
func FuzzZipFutureCompletion(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// yieldPattern
		return []any{seedByte(i, 0)}
	})

	f.Fuzz(func(t *testing.T, yieldPattern uint8) {
		for _, variant := range zipVariants() {
			variant := variant

			t.Run(variant.name, func(t *testing.T) {
				release := make(chan struct{})
				futures := make([]ro.Observable[int], variant.arity)

				for index := range futures {
					index := index
					futures[index] = ro.Future(func() (int, error) {
						<-release
						yieldAt(int64(yieldPattern), index)

						return index, nil
					})
				}

				// Collect blocks until completion, so the futures are released concurrently.
				go func() {
					yieldAt(int64(yieldPattern), variant.arity)
					close(release)
				}()

				var groups [][]int

				var err error

				runWithinDeadline(t, func() { groups, err = ro.Collect(variant.zip(futures)) })

				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}

				if want := [][]int{sequence(0, variant.arity)}; !reflect.DeepEqual(groups, want) {
					t.Fatalf("groups = %v, want %v", groups, want)
				}
			})
		}
	})
}
