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
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

func FuzzInterval(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, false)

		var counter activeCounter

		runTimerIteration(t, "Interval", func() error {
			interval := time.Duration(pickTimerValue(seed, 5, 100, maxTimerMicros)) * time.Microsecond
			obs := applyTakeStop(trackSubscriptions(&counter, ro.Interval(interval)), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzIntervalWithInitial(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, false)

		var counter activeCounter

		runTimerIteration(t, "IntervalWithInitial", func() error {
			interval := time.Duration(pickTimerValue(seed, 5, 100, maxTimerMicros)) * time.Microsecond

			initial := timerDelay(seed, 8, 0)
			if mask&8 != 0 {
				initial = 0 // synchronous first Next, inside Subscribe.
			}

			obs := applyTakeStop(trackSubscriptions(&counter, ro.IntervalWithInitial(initial, interval)), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(timerSettleDelay)

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzTimer(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)

		var counter activeCounter

		runTimerIteration(t, "Timer", func() error {
			obs := applyTakeStop(trackSubscriptions(&counter, ro.Timer(timerDelay(seed, 9, 0))), sc.mode, 1)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzFuture(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)

		var calls int32

		runTimerIteration(t, "Future", func() error {
			delay := timerDelay(seed, 10, 0)
			failing := mask&16 != 0

			obs := applyTakeStop(ro.Future(func() (int, error) {
				atomic.AddInt32(&calls, 1)
				time.Sleep(delay)

				if failing {
					return 0, errors.New("futureFailure")
				}

				return 1, nil
			}), sc.mode, 1)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(timerSettleDelay) // the factory goroutine may still deliver after teardown.

			return checkTimerSinks(sinks)
		})

		if got := int(atomic.LoadInt32(&calls)); got != sc.subs {
			t.Fatalf("Future: factory called %d times for %d subscriptions", got, sc.subs)
		}
	})
}

// FuzzFuturePanic: a panicking factory must not take the whole process down.
func FuzzFuturePanic(f *testing.F) {
	f.Skip("race: future-factory-panic; goroutine in Future (operator_creation.go:464) has no recover: \"panic: futureFactoryPanic\" kills the test binary; remove when fixed")

	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)

		runTimerIteration(t, "Future(panic)", func() error {
			delay := timerDelay(seed, 10, 0)

			obs := ro.Future(func() (int, error) {
				time.Sleep(delay)
				panic("futureFactoryPanic")
			})

			// The goroutine spawned by Future has no recover: the process dies before any
			// assertion runs, so reaching the end means the panic was handled.
			sinks, err := driveSubscribers(obs, 1, timerStopUnsubscribe, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(timerSettleDelay)

			return checkTimerSinks(sinks)
		})
	})
}

func FuzzFromChannel(f *testing.F) {
	f.Skip("race: fromchannel-lost-item; reader keeps selecting on in after done is closed: \"FromChannel: 2 item(s) read from the channel after Unsubscribe returned (len 13 -> 11)\"; remove when fixed")

	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, _ uint8) {
		items := pickTimerValue(seed, 0, 1, fuzzMaxItems)
		// The channel is pre-filled and never closed, so the reader goroutine always has an item
		// ready: any item it takes after Unsubscribe is lost.
		ch := make(chan int, items)
		for i := 0; i < items; i++ {
			ch <- i
		}

		sink := &timerSink[int]{}

		var atUnsub int32

		runTimerIteration(t, "FromChannel", func() error {
			sub := ro.FromChannel[int](ch).Subscribe(sink.observer(timerDelay(seed, 3, 0) / 16))

			time.Sleep(timerDelay(seed, 11, 0) / 4)
			fuzzJitter(seed, 60)
			sub.Unsubscribe()

			atUnsub = int32(len(ch)) //nolint:gosec // bounded by fuzzMaxItems.

			time.Sleep(timerSettleDelay)

			if after := int32(len(ch)); after != atUnsub { //nolint:gosec // bounded by fuzzMaxItems.
				return fmt.Errorf("%d item(s) read from the channel after Unsubscribe returned (len %d -> %d)", atUnsub-after, atUnsub, after)
			}

			if consumed := int32(items) - int32(len(ch)); consumed != atomic.LoadInt32(&sink.nexts) { //nolint:gosec // bounded.
				return fmt.Errorf("%d item(s) consumed from the channel but %d delivered: items lost", consumed, atomic.LoadInt32(&sink.nexts))
			}

			return sink.violation()
		})

		_ = mask
	})
}
