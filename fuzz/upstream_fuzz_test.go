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
	"errors"
	"fmt"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// upstreamInputs holds the tracked sources an operator may subscribe to.
type upstreamInputs struct {
	main     ro.Observable[int]
	signal   ro.Observable[int]
	fallback ro.Observable[int]
	k        int
}

// terminalCounter records downstream terminal notifications.
type terminalCounter struct{ terminals int32 }

// upstreamCase wires one operator over the tracked sources of an iteration.
type upstreamCase struct {
	name string
	run  func(ctx context.Context, in upstreamInputs, term *terminalCounter) ro.Subscription
}

func newUpstreamCase[R any](name string, build func(in upstreamInputs) ro.Observable[R]) upstreamCase {
	return upstreamCase{
		name: name,
		run: func(ctx context.Context, in upstreamInputs, term *terminalCounter) ro.Subscription {
			return build(in).SubscribeWithContext(ctx, ro.NewObserverWithContext(
				func(_ context.Context, _ R) {},
				func(_ context.Context, _ error) { atomic.AddInt32(&term.terminals, 1) },
				func(_ context.Context) { atomic.AddInt32(&term.terminals, 1) },
			))
		},
	}
}

func upstreamCases() []upstreamCase {
	return []upstreamCase{
		newUpstreamCase("Map", func(in upstreamInputs) ro.Observable[int] {
			return ro.Map(func(v int) int { return v + 1 })(in.main)
		}),
		newUpstreamCase("Filter", func(in upstreamInputs) ro.Observable[int] {
			return ro.Filter(func(v int) bool { return v%2 == 0 })(in.main)
		}),
		newUpstreamCase("Take", func(in upstreamInputs) ro.Observable[int] { return ro.Take[int](int64(in.k))(in.main) }),
		newUpstreamCase("Skip", func(in upstreamInputs) ro.Observable[int] { return ro.Skip[int](int64(in.k))(in.main) }),
		newUpstreamCase("TakeWhile", func(in upstreamInputs) ro.Observable[int] {
			return ro.TakeWhile(func(v int) bool { return v < in.k })(in.main)
		}),
		newUpstreamCase("TakeLast", func(in upstreamInputs) ro.Observable[int] { return ro.TakeLast[int](in.k)(in.main) }),
		newUpstreamCase("SkipLast", func(in upstreamInputs) ro.Observable[int] { return ro.SkipLast[int](in.k)(in.main) }),
		newUpstreamCase("SkipWhile", func(in upstreamInputs) ro.Observable[int] {
			return ro.SkipWhile(func(v int) bool { return v < in.k })(in.main)
		}),
		newUpstreamCase("Scan", func(in upstreamInputs) ro.Observable[int] {
			return ro.Scan(func(acc, v int) int { return acc + v }, 0)(in.main)
		}),
		newUpstreamCase("Distinct", func(in upstreamInputs) ro.Observable[int] { return ro.Distinct[int]()(in.main) }),
		newUpstreamCase("Tap", func(in upstreamInputs) ro.Observable[int] {
			return ro.Tap(func(int) {}, func(error) {}, func() {})(in.main)
		}),
		newUpstreamCase("Materialize", func(in upstreamInputs) ro.Observable[ro.Notification[int]] {
			return ro.Materialize[int]()(in.main)
		}),
		newUpstreamCase("Timestamp", func(in upstreamInputs) ro.Observable[ro.TimestampValue[int]] {
			return ro.Timestamp[int]()(in.main)
		}),
		newUpstreamCase("Head", func(in upstreamInputs) ro.Observable[int] { return ro.Head[int]()(in.main) }),
		newUpstreamCase("First", func(in upstreamInputs) ro.Observable[int] {
			return ro.First(func(v int) bool { return v >= in.k })(in.main)
		}),
		newUpstreamCase("ElementAt", func(in upstreamInputs) ro.Observable[int] { return ro.ElementAt[int](in.k)(in.main) }),
		newUpstreamCase("TakeUntil", func(in upstreamInputs) ro.Observable[int] { return ro.TakeUntil[int, int](in.signal)(in.main) }),
		newUpstreamCase("SkipUntil", func(in upstreamInputs) ro.Observable[int] { return ro.SkipUntil[int, int](in.signal)(in.main) }),
		newUpstreamCase("StartWith", func(in upstreamInputs) ro.Observable[int] { return ro.StartWith(-1, -2)(in.main) }),
		newUpstreamCase("EndWith", func(in upstreamInputs) ro.Observable[int] { return ro.EndWith(-1, -2)(in.main) }),
		newUpstreamCase("Pairwise", func(in upstreamInputs) ro.Observable[[]int] { return ro.Pairwise[int]()(in.main) }),
		newUpstreamCase("DefaultIfEmpty", func(in upstreamInputs) ro.Observable[int] { return ro.DefaultIfEmpty(-1)(in.main) }),
		newUpstreamCase("ThrowIfEmpty", func(in upstreamInputs) ro.Observable[int] {
			return ro.ThrowIfEmpty[int](func() error { return errShortCircuitBoom })(in.main)
		}),
		newUpstreamCase("Catch", func(in upstreamInputs) ro.Observable[int] {
			return ro.Catch(func(error) ro.Observable[int] { return in.fallback })(in.main)
		}),
		newUpstreamCase("OnErrorReturn", func(in upstreamInputs) ro.Observable[int] { return ro.OnErrorReturn(-1)(in.main) }),
	}
}

// FuzzUpstreamPropagation asserts that every tracked upstream (source, signal, fallback) is unsubscribed
// once the downstream completed, errored or unsubscribed.
func FuzzUpstreamPropagation(f *testing.F) {
	addShortCircuitSeeds(f)

	table := upstreamCases()

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		c := table[fuzzBound(seed, 0, len(table)-1)]
		n := pickShortCircuitValue(seed, 0, 0, fuzzMaxItems)
		k := pickShortCircuitValue(seed, 1, 0, n+1)
		mode := pickShortCircuitValue(seed, 5, 0, upstreamStopModeCount-1)
		signalItems := pickShortCircuitValue(seed, 7, 0, 2)
		fallbackItems := pickShortCircuitValue(seed, 8, 0, 4)
		yields := pickShortCircuitValue(seed, 6, 0, maxSchedulerYields)

		var main, signal, fallback activeCounter

		in := upstreamInputs{
			main:     trackSubscriptions(&main, shortCircuitSource(seed, n, fuzzIsAsync(mask, 0), true, mode == upstreamStopError, nil)),
			signal:   trackSubscriptions(&signal, fuzzSource(seed, signalItems, fuzzIsAsync(mask, 1))),
			fallback: trackSubscriptions(&fallback, fuzzSource(seed, fallbackItems, fuzzIsAsync(mask, 2))),
			k:        k,
		}

		name := fmt.Sprintf("%s (async=%v/%v/%v mode=%d)", c.name, fuzzIsAsync(mask, 0), fuzzIsAsync(mask, 1), fuzzIsAsync(mask, 2), mode)
		term := &terminalCounter{}

		runShortCircuitIteration(t, name, func() error {
			sub := c.run(context.Background(), in, term)

			if mode == upstreamStopUnsubscribe {
				for i := 0; i < yields; i++ {
					runtime.Gosched()
				}
			} else {
				deadline := time.Now().Add(fuzzDeadline / 2)
				for atomic.LoadInt32(&term.terminals) == 0 {
					if time.Now().After(deadline) {
						return errors.New("downstream never terminated")
					}

					time.Sleep(time.Millisecond)
				}
			}

			sub.Unsubscribe()

			counters := []struct {
				label string
				c     *activeCounter
			}{{"main", &main}, {"signal", &signal}, {"fallback", &fallback}}

			for _, cnt := range counters {
				deadline := time.Now().Add(fuzzDeadline / 4)
				for cnt.c.activeCount() != 0 {
					if time.Now().After(deadline) {
						return fmt.Errorf("%s source still has %d active subscription(s) (opened %d) after terminate/unsubscribe",
							cnt.label, cnt.c.activeCount(), cnt.c.totalCount())
					}

					time.Sleep(time.Millisecond)
				}
			}

			return nil
		})
	})
}
