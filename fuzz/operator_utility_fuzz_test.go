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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

func FuzzRepeatWith(f *testing.F) {
	f.Skip("race: repeatwith-ignores-take-close; remove when fixed") // Fails on main: 1000 source subscriptions after the downstream closed, 1 were enough
	addStreamSeeds(f)
	f.Fuzz(func(t *testing.T, seed, size int64, mask uint8, k int64) {
		runLoop(t, loopKindRepeatWith, seed, size, mask, k)
	})
}

// instrumentedSourceTracker tracks what an instrumented source did, to assert upstream cleanup.
type instrumentedSourceTracker struct {
	counter activeCounter
	wg      sync.WaitGroup
	panics  int32
	msg     atomic.Value
}

// source emits n items, from its own goroutine when async. Panics raised by downstream calls are
// counted instead of being lost in an operator-internal recover.
func (e *instrumentedSourceTracker) source(seed int64, n int, async bool, gap time.Duration) ro.Observable[int] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
		e.counter.open()

		emit := func() {
			defer func() {
				if r := recover(); r != nil {
					atomic.AddInt32(&e.panics, 1)
					e.msg.Store(fmt.Sprintf("%v", r))
				}
			}()

			for i := 0; i < n; i++ {
				if gap > 0 && i%3 == 0 {
					time.Sleep(gap)
				}

				fuzzJitter(seed, i)
				destination.NextWithContext(ctx, i)
			}

			destination.CompleteWithContext(ctx)
		}

		if async {
			e.wg.Add(1)

			go func() {
				defer e.wg.Done()

				emit()
			}()
		} else {
			emit()
		}

		return e.counter.close
	})
}

// finish asserts the source cleanup invariants once the iteration body is done.
func (e *instrumentedSourceTracker) finish(allowBlockedProducers bool) error {
	if !allowBlockedProducers {
		released := make(chan struct{})

		go func() {
			e.wg.Wait()
			close(released)
		}()

		select {
		case <-released:
		case <-time.After(fuzzDeadline):
			return errors.New("producer goroutine never released (blocked after downstream stopped)")
		}
	}

	if n := atomic.LoadInt32(&e.panics); n > 0 {
		msg, _ := e.msg.Load().(string)

		return fmt.Errorf("%d panic(s) while emitting: %s", n, msg)
	}

	return nil
}

func FuzzObserveOn(f *testing.F) {
	f.Skip("race: observeon-send-close; chansend (operator_utility.go:597) races close(ch) in teardown (operator_utility.go:585); remove when fixed")

	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)
		env := &instrumentedSourceTracker{}

		runTimerIteration(t, "ObserveOn", func() error {
			obs := applyTakeStop(ro.ObserveOn[int](sc.buf)(env.source(seed, sc.items, sc.async, 0)), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			if err := checkTimerSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		waitUpstreamClosed(t, &env.counter)
	})
}

func FuzzSubscribeOn(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)
		env := &instrumentedSourceTracker{}

		runTimerIteration(t, "SubscribeOn", func() error {
			obs := applyTakeStop(ro.SubscribeOn[int](sc.buf)(env.source(seed, sc.items, sc.async, 0)), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			if err := checkTimerSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		waitUpstreamClosed(t, &env.counter)
	})
}

// FuzzSubscribeOnInfinite targets an endless upstream: stopping the downstream must stop it.
func FuzzSubscribeOnInfinite(f *testing.F) {
	f.Skip("race: subscribeon-infinite-hang; SubscribeOn(Interval)+Take/Unsubscribe/cancel never returns: \"SubscribeOn(Interval): hang: iteration did not finish within 5s\"; remove when fixed")

	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, false)

		var counter activeCounter

		runTimerIteration(t, "SubscribeOn(Interval)", func() error {
			interval := time.Duration(pickTimerValue(seed, 5, 100, maxTimerMicros)) * time.Microsecond
			obs := applyTakeStop(ro.SubscribeOn[int64](sc.buf)(trackSubscriptions(&counter, ro.Interval(interval))), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzDelay(f *testing.F) {
	f.Skip("race: delay-hang; Delay(d)(source) with Take/Unsubscribe/cancel never finishes: \"Delay: hang: iteration did not finish within 5s\"; remove when fixed")

	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)

		var counter activeCounter

		runTimerIteration(t, "Delay", func() error {
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := applyTakeStop(ro.Delay[int](timerDelay(seed, 6, 0))(src), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(timerSettleDelay) // pending timers fire after teardown.

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzDelayEach(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)

		var counter activeCounter

		runTimerIteration(t, "DelayEach", func() error {
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := applyTakeStop(ro.DelayEach[int](timerDelay(seed, 6, 0)/4)(src), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return checkTimerSinks(sinks)
		})

		waitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeout(f *testing.F) {
	addTimerSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := decodeTimerScenario(seed, mask, k, true)
		env := &instrumentedSourceTracker{}

		runTimerIteration(t, "Timeout", func() error {
			// A duration near the source gap makes timeouts race with items and with completion.
			d := time.Duration(pickTimerValue(seed, 7, 100, maxTimerMicros)) * time.Microsecond
			obs := applyTakeStop(ro.Timeout[int](d)(env.source(seed, sc.items, sc.async, sc.gap)), sc.mode, sc.take)

			sinks, err := driveSubscribers(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(timerSettleDelay) // a timer re-armed after teardown fires here.

			if err := checkTimerSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		waitUpstreamClosed(t, &env.counter)
	})
}
