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
	"testing"

	"github.com/samber/ro"
)

const (
	// maxSignalItems and maxFallbackItems bound the items of the secondary sources. A signal source
	// emitting nothing, one or two items covers "never fires", "fires once" and "fires again".
	maxSignalItems   = 2
	maxFallbackItems = 4

	// maxYieldsBeforeUnsubscribe bounds the scheduler yields before the downstream unsubscribes.
	maxYieldsBeforeUnsubscribe = 50
)

// upstreamSources are the counted sources an operator may subscribe to. mainSource is the one the
// operator is applied to, signalSource is the notifier of TakeUntil and SkipUntil, fallbackSource is the
// replacement of Catch. threshold is the argument of the operators that take a count or a bound.
type upstreamSources struct {
	mainSource     ro.Observable[int]
	signalSource   ro.Observable[int]
	fallbackSource ro.Observable[int]
	threshold      int
}

// upstreamOperator is one row of the table: the operator name, shown in failures, and a function that
// applies it to the sources and subscribes a recorder to the result.
type upstreamOperator struct {
	name      string
	subscribe func(ctx context.Context, sources upstreamSources) (ro.Subscription, progress)
}

func newUpstreamOperator[R any](name string, apply func(sources upstreamSources) ro.Observable[R]) upstreamOperator {
	return upstreamOperator{
		name: name,
		subscribe: func(ctx context.Context, sources upstreamSources) (ro.Subscription, progress) {
			observed := newRecorder[R]()

			return apply(sources).SubscribeWithContext(ctx, observed), observed
		},
	}
}

// upstreamOperators lists the operators that must release every upstream they subscribed to.
func upstreamOperators() []upstreamOperator {
	return []upstreamOperator{
		newUpstreamOperator("Map", func(s upstreamSources) ro.Observable[int] {
			return ro.Map(func(item int) int { return item + 1 })(s.mainSource)
		}),
		newUpstreamOperator("Filter", func(s upstreamSources) ro.Observable[int] {
			return ro.Filter(func(item int) bool { return item%2 == 0 })(s.mainSource)
		}),
		newUpstreamOperator("Take", func(s upstreamSources) ro.Observable[int] { return ro.Take[int](int64(s.threshold))(s.mainSource) }),
		newUpstreamOperator("Skip", func(s upstreamSources) ro.Observable[int] { return ro.Skip[int](int64(s.threshold))(s.mainSource) }),
		newUpstreamOperator("TakeWhile", func(s upstreamSources) ro.Observable[int] {
			return ro.TakeWhile(func(item int) bool { return item < s.threshold })(s.mainSource)
		}),
		newUpstreamOperator("TakeLast", func(s upstreamSources) ro.Observable[int] { return ro.TakeLast[int](s.threshold)(s.mainSource) }),
		newUpstreamOperator("SkipLast", func(s upstreamSources) ro.Observable[int] { return ro.SkipLast[int](s.threshold)(s.mainSource) }),
		newUpstreamOperator("SkipWhile", func(s upstreamSources) ro.Observable[int] {
			return ro.SkipWhile(func(item int) bool { return item < s.threshold })(s.mainSource)
		}),
		newUpstreamOperator("Scan", func(s upstreamSources) ro.Observable[int] {
			return ro.Scan(func(sum, item int) int { return sum + item }, 0)(s.mainSource)
		}),
		newUpstreamOperator("Distinct", func(s upstreamSources) ro.Observable[int] { return ro.Distinct[int]()(s.mainSource) }),
		newUpstreamOperator("Tap", func(s upstreamSources) ro.Observable[int] {
			return ro.Tap(func(int) {}, func(error) {}, func() {})(s.mainSource)
		}),
		newUpstreamOperator("Materialize", func(s upstreamSources) ro.Observable[ro.Notification[int]] {
			return ro.Materialize[int]()(s.mainSource)
		}),
		newUpstreamOperator("Timestamp", func(s upstreamSources) ro.Observable[ro.TimestampValue[int]] {
			return ro.Timestamp[int]()(s.mainSource)
		}),
		newUpstreamOperator("Head", func(s upstreamSources) ro.Observable[int] { return ro.Head[int]()(s.mainSource) }),
		newUpstreamOperator("First", func(s upstreamSources) ro.Observable[int] {
			return ro.First(func(item int) bool { return item >= s.threshold })(s.mainSource)
		}),
		newUpstreamOperator("ElementAt", func(s upstreamSources) ro.Observable[int] { return ro.ElementAt[int](s.threshold)(s.mainSource) }),
		newUpstreamOperator("TakeUntil", func(s upstreamSources) ro.Observable[int] {
			return ro.TakeUntil[int, int](s.signalSource)(s.mainSource)
		}),
		newUpstreamOperator("SkipUntil", func(s upstreamSources) ro.Observable[int] {
			return ro.SkipUntil[int, int](s.signalSource)(s.mainSource)
		}),
		newUpstreamOperator("StartWith", func(s upstreamSources) ro.Observable[int] { return ro.StartWith(-1, -2)(s.mainSource) }),
		newUpstreamOperator("EndWith", func(s upstreamSources) ro.Observable[int] { return ro.EndWith(-1, -2)(s.mainSource) }),
		newUpstreamOperator("Pairwise", func(s upstreamSources) ro.Observable[[]int] { return ro.Pairwise[int]()(s.mainSource) }),
		newUpstreamOperator("DefaultIfEmpty", func(s upstreamSources) ro.Observable[int] { return ro.DefaultIfEmpty(-1)(s.mainSource) }),
		newUpstreamOperator("ThrowIfEmpty", func(s upstreamSources) ro.Observable[int] {
			return ro.ThrowIfEmpty[int](func() error { return errInjectedFailure })(s.mainSource)
		}),
		newUpstreamOperator("Catch", func(s upstreamSources) ro.Observable[int] {
			return ro.Catch(func(error) ro.Observable[int] { return s.fallbackSource })(s.mainSource)
		}),
		newUpstreamOperator("OnErrorReturn", func(s upstreamSources) ro.Observable[int] { return ro.OnErrorReturn(-1)(s.mainSource) }),
	}
}

// upstreamScenario is what a fuzz input says about one iteration: which operator, and the sources
// it is applied to.
type upstreamScenario struct {
	operator upstreamOperator
	main     *source
	signal   *source
	fallback *source

	mainCounter, signalCounter, fallbackCounter subscriptionCounter
	threshold                                   int
}

func newUpstreamScenario(operatorIndex, items, threshold, signalItems, fallbackItems uint8, asyncMain, asyncSignal, asyncFallback, mainFails bool) *upstreamScenario {
	operators := upstreamOperators()
	count := bounded(items, 0, maxItems)

	scenario := &upstreamScenario{
		operator:  operators[bounded(operatorIndex, 0, len(operators)-1)],
		main:      newSource(count, asyncMain),
		signal:    newSource(bounded(signalItems, 0, maxSignalItems), asyncSignal),
		fallback:  newSource(bounded(fallbackItems, 0, maxFallbackItems), asyncFallback),
		threshold: bounded(threshold, 0, count+1), // count+1: a threshold beyond the last item
	}

	if mainFails {
		scenario.main.failingAtEnd(errInjectedFailure)
	}

	return scenario
}

func (s *upstreamScenario) sources() upstreamSources {
	return upstreamSources{
		mainSource:     countSubscriptions(&s.mainCounter, s.main.observable()),
		signalSource:   countSubscriptions(&s.signalCounter, s.signal.observable()),
		fallbackSource: countSubscriptions(&s.fallbackCounter, s.fallback.observable()),
		threshold:      s.threshold,
	}
}

// subscribe applies the operator to the sources and subscribes to the result. A Subscribe that does not
// return fails the test.
func (s *upstreamScenario) subscribe(t *testing.T) (ro.Subscription, progress) {
	t.Helper()

	var (
		subscription ro.Subscription
		observed     progress
	)

	runWithinDeadline(t, func() { subscription, observed = s.operator.subscribe(context.Background(), s.sources()) })

	return subscription, observed
}

// expectEveryUpstreamReleased waits until the main, signal and fallback sources have no live subscription.
func (s *upstreamScenario) expectEveryUpstreamReleased(t *testing.T) {
	t.Helper()

	for label, counter := range map[string]*subscriptionCounter{"main": &s.mainCounter, "signal": &s.signalCounter, "fallback": &s.fallbackCounter} {
		waitUntil(t, label+" source (operator "+s.operator.name+") to be released", func() bool { return counter.activeCount() == 0 })
	}
}

// FuzzUpstreamReleasedOnCompletion applies one of 25 operators to a main source that completes, with a
// signal source and a fallback source on the side. Each of the three sources is synchronous or asynchronous.
//
// Invariant: once the downstream completed and the test unsubscribed, no source keeps a live subscription.
//
// Seeds: operatorIndex walks the table; the numeric inputs spread over their range; the three async flags
// cycle through every combination.
func FuzzUpstreamReleasedOnCompletion(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// operatorIndex, items, threshold, signalItems, fallbackItems, asyncMain, asyncSignal, asyncFallback
		return []any{uint8(i), seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%4 < 2, i%8 < 4}
	})

	f.Fuzz(func(t *testing.T, operatorIndex, items, threshold, signalItems, fallbackItems uint8, asyncMain, asyncSignal, asyncFallback bool) {
		scenario := newUpstreamScenario(operatorIndex, items, threshold, signalItems, fallbackItems, asyncMain, asyncSignal, asyncFallback, false)

		subscription, observed := scenario.subscribe(t)
		waitUntil(t, "the downstream terminal notification", func() bool { return observed.terminalCount() > 0 })
		subscription.Unsubscribe()

		scenario.expectEveryUpstreamReleased(t)
	})
}

// FuzzUpstreamReleasedOnError is FuzzUpstreamReleasedOnCompletion with a main source that fails at its end.
//
// Invariant: once the downstream terminated (with an error, or by recovering from it) and the test
// unsubscribed, no source keeps a live subscription.
//
// Seeds: as FuzzUpstreamReleasedOnCompletion.
func FuzzUpstreamReleasedOnError(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// operatorIndex, items, threshold, signalItems, fallbackItems, asyncMain, asyncSignal, asyncFallback
		return []any{uint8(i), seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), i%2 == 0, i%4 < 2, i%8 < 4}
	})

	f.Fuzz(func(t *testing.T, operatorIndex, items, threshold, signalItems, fallbackItems uint8, asyncMain, asyncSignal, asyncFallback bool) {
		scenario := newUpstreamScenario(operatorIndex, items, threshold, signalItems, fallbackItems, asyncMain, asyncSignal, asyncFallback, true)

		subscription, observed := scenario.subscribe(t)
		waitUntil(t, "the downstream terminal notification", func() bool { return observed.terminalCount() > 0 })
		subscription.Unsubscribe()

		scenario.expectEveryUpstreamReleased(t)
	})
}

// FuzzUpstreamReleasedOnUnsubscribe is FuzzUpstreamReleasedOnCompletion with an Unsubscribe from the test
// after a few scheduler yields, whatever the stream did before.
//
// Invariant: no source keeps a live subscription after the Unsubscribe.
//
// Seeds: as FuzzUpstreamReleasedOnCompletion; yields spreads over its range.
func FuzzUpstreamReleasedOnUnsubscribe(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// operatorIndex, items, threshold, signalItems, fallbackItems, yields, asyncMain, asyncSignal, asyncFallback
		return []any{uint8(i), seedByte(i, 0), seedByte(i, 1), seedByte(i, 2), seedByte(i, 3), seedByte(i, 4), i%2 == 0, i%4 < 2, i%8 < 4}
	})

	f.Fuzz(func(t *testing.T, operatorIndex, items, threshold, signalItems, fallbackItems, yields uint8, asyncMain, asyncSignal, asyncFallback bool) {
		scenario := newUpstreamScenario(operatorIndex, items, threshold, signalItems, fallbackItems, asyncMain, asyncSignal, asyncFallback, false)

		subscription, _ := scenario.subscribe(t)
		for yield := 0; yield < bounded(yields, 0, maxYieldsBeforeUnsubscribe); yield++ {
			runtime.Gosched()
		}

		subscription.Unsubscribe()

		scenario.expectEveryUpstreamReleased(t)
	})
}
