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

// catchSources is a source that fails, and the two sources that ro.Catch merges to replace it. Each
// source is synchronous or asynchronous, and each is counted.
type catchSources struct {
	failing, mergedA, mergedB *source

	failingCounter, counterA, counterB subscriptionCounter
}

func newCatchSources(failingItems, itemsA, itemsB uint8, asyncFailing, asyncA, asyncB bool) *catchSources {
	return &catchSources{
		failing: newSource(bounded(failingItems, 0, 3), asyncFailing).failingAtEnd(errInjectedFailure),
		mergedA: newSource(bounded(itemsA, 1, 4), asyncA),
		mergedB: newSource(bounded(itemsB, 1, 4), asyncB),
	}
}

func (s *catchSources) pipeline() ro.Observable[int] {
	merged := ro.Merge(
		countSubscriptions(&s.counterA, s.mergedA.observable()),
		countSubscriptions(&s.counterB, s.mergedB.observable()),
	)

	return ro.Catch(func(error) ro.Observable[int] { return merged })(countSubscriptions(&s.failingCounter, s.failing.observable()))
}

func (s *catchSources) counters() []*subscriptionCounter {
	return []*subscriptionCounter{&s.failingCounter, &s.counterA, &s.counterB}
}

// totalItems is every item the pipeline can deliver: the ones of the failing source, then the merged ones.
func (s *catchSources) totalItems() int { return s.failing.items + s.mergedA.items + s.mergedB.items }

// FuzzCatchRunsToCompletion replaces a failing source of `failingItems` integers with a merge of two
// sources of `itemsA` and `itemsB` integers, and lets the stream end. Each of the three sources is
// synchronous or asynchronous.
//
// Invariant: every item is delivered, the stream completes once without overlapping notifications (the
// merge receives the downstream as it is), and the three source subscriptions are released.
//
// Seeds: item counts spread over their range; the three async flags cycle through every combination.
func FuzzCatchRunsToCompletion(f *testing.F) {
	f.Skip("race: catch-unsafe-merge; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, i%8 < 4}
	})

	f.Fuzz(func(t *testing.T, failingItems, itemsA, itemsB uint8, asyncFailing, asyncA, asyncB bool) {
		sources := newCatchSources(failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB)

		got := runToCompletion(t, sources.pipeline(), sources.counters()...)

		got.expectCompletedOnce(t)
		expectValueCount(t, got, sources.totalItems())
	})
}

// FuzzCatchStoppedByTake is FuzzCatchRunsToCompletion with the stream closed from inside the pipeline by
// ro.Take(takeCount).
//
// Invariant: exactly min(takeCount, total items) values are delivered, nothing overlaps, and the three
// source subscriptions are released.
//
// Seeds: as FuzzCatchRunsToCompletion; takeCount spreads over its range.
func FuzzCatchStoppedByTake(f *testing.F) {
	f.Skip("race: catch-unsafe-merge; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB, takeCount
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, i%8 < 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, failingItems, itemsA, itemsB uint8, asyncFailing, asyncA, asyncB bool, takeCount uint8) {
		sources := newCatchSources(failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB)
		taken := bounded(takeCount, 1, maxLoopTake)

		got := runWithTake(t, sources.pipeline(), taken, sources.counters()...)

		expectValueCount(t, got, smaller(taken, sources.totalItems()))
	})
}

// FuzzCatchStoppedByUnsubscribe is FuzzCatchRunsToCompletion with an Unsubscribe from outside once
// `unsubscribeAfter` values arrived.
//
// Invariant: nothing overlaps, no notification follows a terminal one, and the three source subscriptions
// are released.
//
// Seeds: as FuzzCatchRunsToCompletion; unsubscribeAfter spreads over its range.
func FuzzCatchStoppedByUnsubscribe(f *testing.F) {
	f.Skip("race: catch-unsafe-merge; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB, unsubscribeAfter
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2, i%8 < 4, seedByte(i, 3)}
	})

	f.Fuzz(func(t *testing.T, failingItems, itemsA, itemsB uint8, asyncFailing, asyncA, asyncB bool, unsubscribeAfter uint8) {
		sources := newCatchSources(failingItems, itemsA, itemsB, asyncFailing, asyncA, asyncB)

		runThenUnsubscribe(t, sources.pipeline(), bounded(unsubscribeAfter, 1, maxLoopTake), sources.counters()...)
	})
}

// FuzzWhileStoppedByTake repeats a source of `items` integers while an endless condition holds, and
// closes the stream with ro.Take(takeCount).
//
// Invariant: exactly takeCount items are delivered, and no run is opened after the downstream closed.
// The loop must stop on its own: Subscribe returns.
//
// Seeds: items and takeCount spread over their range; asyncSource alternates.
func FuzzWhileStoppedByTake(f *testing.F) {
	f.Skip("race: while-ignores-closed-destination; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, takeCount, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, takeCount uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		taken := bounded(takeCount, 1, maxLoopTake)
		loop := ro.While[int](newLoopCondition(t, unlimitedLoop).holds)

		got, upstream := runLoopStoppedByTake(t, loop, newSource(itemsPerRun, asyncSource), taken)

		expectLoopStoppedByTake(t, got, upstream, taken, itemsPerRun)
	})
}

// FuzzWhileFinite repeats a source of `items` integers while a condition holds `rounds` times.
//
// Invariant: the source is subscribed `rounds` times, every run delivers its items, and the stream completes once.
//
// Seeds: items and rounds spread over their range; asyncSource alternates.
func FuzzWhileFinite(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, rounds, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, rounds uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		runs := bounded(rounds, 1, maxFiniteLoopRuns)
		loop := ro.While[int](newLoopCondition(t, runs).holds)

		got, upstream := runFiniteLoop(t, loop, newSource(itemsPerRun, asyncSource))

		expectFiniteLoop(t, got, upstream, runs, itemsPerRun)
	})
}

// FuzzDoWhileStoppedByTake is FuzzWhileStoppedByTake for ro.DoWhile.
//
// Invariant: exactly takeCount items are delivered, and no run is opened after the downstream closed.
//
// Seeds: items and takeCount spread over their range; asyncSource alternates.
func FuzzDoWhileStoppedByTake(f *testing.F) {
	f.Skip("race: dowhile-ignores-closed-destination; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, takeCount, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, takeCount uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		taken := bounded(takeCount, 1, maxLoopTake)
		loop := ro.DoWhile[int](newLoopCondition(t, unlimitedLoop).holds)

		got, upstream := runLoopStoppedByTake(t, loop, newSource(itemsPerRun, asyncSource), taken)

		expectLoopStoppedByTake(t, got, upstream, taken, itemsPerRun)
	})
}

// FuzzDoWhileFinite runs a source of `items` integers once, then again each time a condition that holds
// `rounds` times says so.
//
// Invariant: the source is subscribed rounds+1 times (the first run is unconditional), every run delivers
// its items, and the stream completes once.
//
// Seeds: items and rounds spread over their range; asyncSource alternates.
func FuzzDoWhileFinite(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, rounds, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, rounds uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		conditionRuns := bounded(rounds, 1, maxFiniteLoopRuns)
		loop := ro.DoWhile[int](newLoopCondition(t, conditionRuns).holds)

		got, upstream := runFiniteLoop(t, loop, newSource(itemsPerRun, asyncSource))

		expectFiniteLoop(t, got, upstream, conditionRuns+1, itemsPerRun)
	})
}

// FuzzRetryStoppedByTake retries a source of `items` integers that fails after its last item, and closes
// the stream with ro.Take(takeCount).
//
// Invariant: exactly takeCount items are delivered, and no retry is opened after the downstream closed.
//
// Seeds: items and takeCount spread over their range; asyncSource alternates.
func FuzzRetryStoppedByTake(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, takeCount, asyncSource
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0}
	})

	f.Fuzz(func(t *testing.T, items, takeCount uint8, asyncSource bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		taken := bounded(takeCount, 1, maxLoopTake)
		failing := newSource(itemsPerRun, asyncSource).failingAtEnd(errInjectedFailure)

		got, upstream := runLoopStoppedByTake(t, ro.Retry[int](), failing, taken)

		expectLoopStoppedByTake(t, got, upstream, taken, itemsPerRun)
	})
}

// FuzzRetryWithConfigStoppedByTake retries a source of `items` integers that fails after its last item, at
// most `maxRetries` times, and closes the stream with ro.Take(takeCount). With resetOnSuccess the retry
// budget is restored each time a run delivers an item.
//
// Invariant: when the retries run out before takeCount items, the error reaches the downstream after
// exactly maxRetries+1 runs; otherwise exactly takeCount items are delivered and no retry is opened after
// the downstream closed.
//
// Seeds: numeric inputs spread over their range; asyncSource and resetOnSuccess cycle through every combination.
func FuzzRetryWithConfigStoppedByTake(f *testing.F) {
	f.Skip("race: retry-ignores-closed-destination; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, takeCount, maxRetries, asyncSource, resetOnSuccess
		return []any{seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), i%2 == 0, i%4 < 2}
	})

	f.Fuzz(func(t *testing.T, items, takeCount, maxRetries uint8, asyncSource, resetOnSuccess bool) {
		itemsPerRun := bounded(items, 1, maxLoopItemsPerRun)
		taken := bounded(takeCount, 1, maxLoopTake)
		retries := bounded(maxRetries, 1, maxFiniteLoopRuns)
		failing := newSource(itemsPerRun, asyncSource).failingAtEnd(errInjectedFailure)
		config := ro.RetryConfig{MaxRetries: uint64(retries), ResetOnSuccess: resetOnSuccess} //nolint:gosec // retries is in [1,3].

		got, upstream := runLoopStoppedByTake(t, ro.RetryWithConfig[int](config), failing, taken)

		if !resetOnSuccess && taken > (retries+1)*itemsPerRun {
			expectRetriesExhausted(t, got, upstream, retries+1, itemsPerRun)

			return
		}

		expectLoopStoppedByTake(t, got, upstream, taken, itemsPerRun)
	})
}

// expectRetriesExhausted checks that every allowed run happened, delivered its items, and that the error
// of the last run reached the downstream.
func expectRetriesExhausted(t *testing.T, got *recorder[int], upstream *subscriptionCounter, runs, itemsPerRun int) {
	t.Helper()

	if subscriptions, delivered := upstream.totalCount(), got.valueCount(); subscriptions != runs || delivered != runs*itemsPerRun {
		t.Fatalf("retries exhausted: %d subscriptions (want %d), %d values (want %d)", subscriptions, runs, delivered, runs*itemsPerRun)
	}

	got.expectFailedOnce(t)
}

// resumeChain is the chain of sources of ro.OnErrorResumeNextWith: source i has 1 to 3 items, ends with a
// failure or a completion, and is synchronous or asynchronous. Every subscription is counted.
type resumeChain struct {
	sources       []*source
	counter       subscriptionCounter
	lastFails     bool
	totalItems    int
	itemsPerChain []int
}

func newResumeChain(sourceCount, itemsPerSource uint8, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails bool) *resumeChain {
	count := bounded(sourceCount, 2, 5)
	chain := &resumeChain{lastFails: lastSourceFails}

	for index := 0; index < count; index++ {
		items := bounded(itemsPerSource+uint8(index), 1, 3) //nolint:gosec // index is below 5.

		async := asyncEven
		if index%2 == 1 {
			async = asyncOdd
		}

		sourceFails := earlySourcesFail
		if index == count-1 {
			sourceFails = lastSourceFails
		}

		next := newSource(items, async)
		if sourceFails {
			next = next.failingAtEnd(errInjectedFailure)
		}

		chain.sources = append(chain.sources, next)
		chain.itemsPerChain = append(chain.itemsPerChain, items)
		chain.totalItems += items
	}

	return chain
}

func (c *resumeChain) pipeline() ro.Observable[int] {
	counted := make([]ro.Observable[int], len(c.sources))
	for index, chained := range c.sources {
		counted[index] = countSubscriptions(&c.counter, chained.observable())
	}

	return ro.OnErrorResumeNextWith(counted[1:]...)(counted[0])
}

// sourcesNeededFor is how many sources of the chain must be subscribed to deliver `items` items.
func (c *resumeChain) sourcesNeededFor(items int) int {
	needed, delivered := 0, 0
	for needed < len(c.sources) && delivered < items {
		delivered += c.itemsPerChain[needed]
		needed++
	}

	return needed
}

// FuzzOnErrorResumeNextWithRunsToCompletion chains 2 to 5 sources: when one ends, by failure or by
// completion, the next one is subscribed. Sources at even and odd positions are synchronous or asynchronous
// independently. Every source before the last one fails when earlySourcesFail, else completes.
//
// Invariant: every source is subscribed, every item delivered, and the terminal notification is the one of
// the last source: an error when lastSourceFails, else a completion.
//
// Seeds: sourceCount and itemsPerSource spread over their range; the four bools cycle through every combination.
func FuzzOnErrorResumeNextWithRunsToCompletion(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// sourceCount, itemsPerSource, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%8 < 4, i%16 < 8}
	})

	f.Fuzz(func(t *testing.T, sourceCount, itemsPerSource uint8, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails bool) {
		chain := newResumeChain(sourceCount, itemsPerSource, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails)

		got := runToCompletion(t, chain.pipeline(), &chain.counter)

		if subscribed := chain.counter.totalCount(); subscribed != len(chain.sources) {
			t.Fatalf("%d sources subscribed, want all %d", subscribed, len(chain.sources))
		}

		expectValueCount(t, got, chain.totalItems)

		if chain.lastFails {
			got.expectFailedOnce(t)
		} else {
			got.expectCompletedOnce(t)
		}
	})
}

// FuzzOnErrorResumeNextWithStoppedByTake is FuzzOnErrorResumeNextWithRunsToCompletion with the stream
// closed from inside the pipeline by ro.Take(takeCount).
//
// Invariant: exactly min(takeCount, total items) values are delivered, and no source is subscribed once
// the sources already subscribed have delivered takeCount items.
//
// Seeds: as FuzzOnErrorResumeNextWithRunsToCompletion; takeCount spreads over its range.
func FuzzOnErrorResumeNextWithStoppedByTake(f *testing.F) {
	f.Skip("race: resumenext-ignores-closed-destination; remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// sourceCount, itemsPerSource, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails, takeCount
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%8 < 4, i%16 < 8, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, sourceCount, itemsPerSource uint8, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails bool, takeCount uint8) {
		chain := newResumeChain(sourceCount, itemsPerSource, asyncEven, asyncOdd, earlySourcesFail, lastSourceFails)
		taken := bounded(takeCount, 1, maxLoopTake)

		got := runWithTake(t, chain.pipeline(), taken, &chain.counter)

		expectValueCount(t, got, smaller(taken, chain.totalItems))

		if subscribed, enough := chain.counter.totalCount(), chain.sourcesNeededFor(taken); subscribed > enough {
			t.Fatalf("%d sources subscribed after the downstream closed, %d were enough", subscribed, enough)
		}
	})
}
