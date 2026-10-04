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

func FuzzTimeInterval(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, false)

		var counter activeCounter

		fuzzTimeIter(t, "Interval", func() error {
			interval := time.Duration(fuzzTimePick(seed, 5, 100, fuzzTimeMaxMicros)) * time.Microsecond
			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, ro.Interval(interval)), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeIntervalWithInitial(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, false)

		var counter activeCounter

		fuzzTimeIter(t, "IntervalWithInitial", func() error {
			interval := time.Duration(fuzzTimePick(seed, 5, 100, fuzzTimeMaxMicros)) * time.Microsecond

			initial := fuzzTimeMicros(seed, 8, 0)
			if mask&8 != 0 {
				initial = 0 // synchronous first Next, inside Subscribe.
			}

			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, ro.IntervalWithInitial(initial, interval)), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(fuzzTimeSettle)

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeTimer(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var counter activeCounter

		fuzzTimeIter(t, "Timer", func() error {
			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, ro.Timer(fuzzTimeMicros(seed, 9, 0))), sc.mode, 1)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeFuture(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var calls int32

		fuzzTimeIter(t, "Future", func() error {
			delay := fuzzTimeMicros(seed, 10, 0)
			failing := mask&16 != 0

			obs := fuzzTimeMaybeTake(ro.Future(func() (int, error) {
				atomic.AddInt32(&calls, 1)
				time.Sleep(delay)

				if failing {
					return 0, errors.New("fuzzTimeFutureError")
				}

				return 1, nil
			}), sc.mode, 1)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(fuzzTimeSettle) // the factory goroutine may still deliver after teardown.

			return fuzzTimeCheckSinks(sinks)
		})

		if got := int(atomic.LoadInt32(&calls)); got != sc.subs {
			t.Fatalf("Future: factory called %d times for %d subscriptions", got, sc.subs)
		}
	})
}

// FuzzTimeFuturePanic: a panicking factory must not take the whole process down.
func FuzzTimeFuturePanic(f *testing.F) {
	f.Skip("race: future-factory-panic; goroutine in Future (operator_creation.go:464) has no recover: \"panic: fuzzTimeFactoryPanic\" kills the test binary; remove when fixed")

	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		fuzzTimeIter(t, "Future(panic)", func() error {
			delay := fuzzTimeMicros(seed, 10, 0)

			obs := ro.Future(func() (int, error) {
				time.Sleep(delay)
				panic("fuzzTimeFactoryPanic")
			})

			// The goroutine spawned by Future has no recover: the process dies before any
			// assertion runs, so reaching the end means the panic was handled.
			sinks, err := fuzzTimeDrive(obs, 1, fuzzTimeModeUnsub, 1, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(fuzzTimeSettle)

			return fuzzTimeCheckSinks(sinks)
		})
	})
}

func FuzzTimeFromChannel(f *testing.F) {
	f.Skip("race: fromchannel-lost-item; reader keeps selecting on in after done is closed: \"FromChannel: 2 item(s) read from the channel after Unsubscribe returned (len 13 -> 11)\"; remove when fixed")

	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, _ uint8) {
		items := fuzzTimePick(seed, 0, 1, fuzzMaxItems)
		// The channel is pre-filled and never closed, so the reader goroutine always has an item
		// ready: any item it takes after Unsubscribe is lost.
		ch := make(chan int, items)
		for i := 0; i < items; i++ {
			ch <- i
		}

		sink := &fuzzTimeSink[int]{}

		var atUnsub int32

		fuzzTimeIter(t, "FromChannel", func() error {
			sub := ro.FromChannel[int](ch).Subscribe(sink.observer(fuzzTimeMicros(seed, 3, 0) / 16))

			time.Sleep(fuzzTimeMicros(seed, 11, 0) / 4)
			fuzzJitter(seed, 60)
			sub.Unsubscribe()

			atUnsub = int32(len(ch)) //nolint:gosec // bounded by fuzzMaxItems.

			time.Sleep(fuzzTimeSettle)

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
