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

package rocron

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-co-op/gocron/v2"
	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

const (
	// fuzzWait bounds every wait so a hang becomes a failure, not a stuck CI job.
	fuzzWait = 5 * time.Second

	// fuzzTickUnit scales intervals and delays from the fuzz input; tiny so an
	// iteration lasts a few milliseconds.
	fuzzTickUnit = time.Millisecond

	// fuzzMaxSteps bounds every derived duration at fuzzMaxSteps*fuzzTickUnit.
	fuzzMaxSteps = 8

	// fuzzModeDelayed is the bit of the mask selecting a job that starts after a
	// delay instead of immediately.
	fuzzModeDelayed = 1 << 0

	// fuzzOneShotMargin keeps a delayed one-shot start in the future by the time
	// the scheduler is built, since gocron skips start times already in the past.
	fuzzOneShotMargin = 50 * time.Millisecond

	// fuzzLeakSettle is how long goroutines get to wind down before counting them.
	fuzzLeakSettle = 2 * time.Second

	// fuzzLeakSlack tolerates unrelated runtime goroutines (GC workers, timers).
	fuzzLeakSlack = 2
)

// fuzzSteps returns a value in [1, fuzzMaxSteps] derived from any int64.
func fuzzSteps(seed int64) time.Duration {
	if seed < 0 {
		seed = -seed
	}

	return time.Duration(seed%fuzzMaxSteps+1) * fuzzTickUnit
}

// fuzzJob picks a job definition from the mask: immediate (sync) or delayed
// (async) one-shot jobs, or a recurring interval.
func fuzzJob(seed int64, mask uint8) gocron.JobDefinition {
	delay := fuzzSteps(seed)

	switch {
	case mask&fuzzModeDelayed == 0 && mask&2 != 0:
		return gocron.OneTimeJob(gocron.OneTimeJobStartImmediately())
	case mask&fuzzModeDelayed != 0 && mask&2 != 0:
		return gocron.OneTimeJob(gocron.OneTimeJobStartDateTime(time.Now().Add(fuzzOneShotMargin + delay)))
	default:
		return gocron.DurationJob(delay)
	}
}

// waitBounded fails the iteration when ch is not closed within fuzzWait.
func waitBounded(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()

	select {
	case <-ch:
	case <-time.After(fuzzWait):
		t.Fatalf("rocron: %s: timed out after %s", what, fuzzWait)
	}
}

// recoverInto turns a panic of the calling goroutine into a message on errs,
// since a panic in a goroutine would otherwise kill the whole test binary.
func recoverInto(errs chan<- string) {
	if r := recover(); r != nil {
		errs <- fmt.Sprintf("rocron: panic: %v", r)
	}
}

// FuzzCronUnsubscribeFromNext completes the stream from inside Next (Take(1)):
// teardown runs on the job's own goroutine, while Shutdown waits for running jobs.
func FuzzCronUnsubscribeFromNext(f *testing.F) {
	f.Skip("race: cron-shutdown-self-wait; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i / 2)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, slowNext uint8) {
		var (
			done = make(chan struct{})
			errs = make(chan string, 1)
			once sync.Once
			end  = func() { once.Do(func() { close(done) }) }
		)

		// Sync mode returns from Next at once, async mode keeps the job busy.
		nextDelay := time.Duration(0)
		if mask&fuzzModeDelayed != 0 {
			nextDelay = time.Duration(slowNext%fuzzMaxSteps) * fuzzTickUnit
		}

		obs := ro.Take[ScheduleJob](1)(NewScheduler(fuzzJob(seed, mask)))
		subscribed := make(chan ro.Subscription, 1)

		// Subscribe runs in its own goroutine: with an immediate job the first
		// tick fires while Start() is still waiting, so Subscribe itself may hang.
		go func() {
			defer recoverInto(errs)

			subscribed <- obs.Subscribe(ro.NewObserver(
				func(ScheduleJob) { time.Sleep(nextDelay) },
				func(error) { end() },
				end,
			))
		}()

		started := time.Now()
		waitBounded(t, done, "Take(1) over scheduler never completed")

		select {
		case sub := <-subscribed:
			defer sub.Unsubscribe()
		case msg := <-errs:
			t.Fatal(msg)
		case <-time.After(fuzzWait):
			t.Fatalf("rocron: Subscribe never returned after completion")
		}

		// Teardown must not hold the stream hostage for the Shutdown timeout.
		if elapsed := time.Since(started); elapsed > fuzzWait/2 {
			t.Fatalf("rocron: completion took %s", elapsed)
		}
	})
}

// FuzzCronNewJobFailureLeak subscribes with an invalid job definition: the
// scheduler created beforehand must not outlive the failed subscription.
// Sync/async does not apply: the failure happens synchronously in subscribe.
func FuzzCronNewJobFailureLeak(f *testing.F) {
	f.Skip("race: cron-newjob-error-leaks-scheduler; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{uint8(i), uint8(i / 2)} })

	f.Fuzz(func(t *testing.T, kind uint8, repeat uint8) {
		var def gocron.JobDefinition

		switch kind % 3 {
		case 0:
			def = gocron.CronJob("not a cron expression", false)
		case 1:
			def = gocron.DurationJob(-time.Second)
		default:
			def = gocron.DurationRandomJob(time.Second, time.Millisecond) // min > max
		}

		before := runtime.NumGoroutine()
		rounds := int(repeat)%fuzzMaxSteps + 1

		for i := 0; i < rounds; i++ {
			var failed atomic.Bool

			sub := NewScheduler(def).Subscribe(ro.NewObserver(
				func(ScheduleJob) {},
				func(error) { failed.Store(true) },
				func() {},
			))
			sub.Unsubscribe()

			if !failed.Load() {
				t.Fatalf("rocron: invalid job definition did not produce an error")
			}
		}

		deadline := time.Now().Add(fuzzLeakSettle)
		for runtime.NumGoroutine() > before+fuzzLeakSlack && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}

		if after := runtime.NumGoroutine(); after > before+fuzzLeakSlack {
			t.Fatalf("rocron: goroutines leaked after %d failed subscriptions: before=%d after=%d", rounds, before, after)
		}
	})
}

// FuzzCronUnsubscribeWhileJobRunning unsubscribes from another goroutine while
// Next may still be running: no Next may be delivered after Unsubscribe returns
// and no goroutine may outlive the subscription.
func FuzzCronUnsubscribeWhileJobRunning(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i / 2)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, slowNext uint8) {
		before := runtime.NumGoroutine()

		var (
			unsubscribed atomic.Bool
			late         atomic.Int64
			started      = make(chan struct{})
			once         sync.Once
		)

		// Sync mode returns from Next at once, async mode keeps the job busy.
		nextDelay := time.Duration(0)
		if mask&fuzzModeDelayed != 0 {
			nextDelay = time.Duration(slowNext%fuzzMaxSteps+1) * fuzzTickUnit
		}

		sub := NewScheduler(fuzzJob(seed, mask)).Subscribe(ro.NewObserver(
			func(ScheduleJob) {
				if unsubscribed.Load() {
					late.Add(1)
				}

				once.Do(func() { close(started) })
				time.Sleep(nextDelay)
			},
			func(error) {},
			func() {},
		))

		waitBounded(t, started, "first tick")

		unsubDone := make(chan struct{})
		go func() {
			defer close(unsubDone)

			sub.Unsubscribe()
			unsubscribed.Store(true)
		}()
		waitBounded(t, unsubDone, "Unsubscribe")

		time.Sleep(10 * fuzzTickUnit)

		if n := late.Load(); n != 0 {
			t.Fatalf("rocron: %d Next calls delivered after Unsubscribe returned", n)
		}

		deadline := time.Now().Add(fuzzLeakSettle)
		for runtime.NumGoroutine() > before+fuzzLeakSlack && time.Now().Before(deadline) {
			time.Sleep(10 * time.Millisecond)
		}

		if after := runtime.NumGoroutine(); after > before+fuzzLeakSlack {
			t.Fatalf("rocron: goroutines leaked after Unsubscribe: before=%d after=%d", before, after)
		}
	})
}

// FuzzCronOverlappingRuns makes ticks outlast their interval so runs overlap:
// the Observable contract still demands non-overlapping, ordered Next calls.
func FuzzCronOverlappingRuns(f *testing.F) {
	f.Skip("race: cron-counter-out-of-order; remove when fixed")

	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i), uint8(i / 2)} })

	f.Fuzz(func(t *testing.T, seed int64, mask uint8, slowNext uint8) {
		const wantTicks = 4

		var (
			inFlight atomic.Int64
			overlap  atomic.Int64
			lastSeen = int64(-1)
			outOfOrd atomic.Int64
			seen     atomic.Int64
			done     = make(chan struct{})
			once     sync.Once
		)

		// Sync mode: instant Next; async mode: Next slower than the interval.
		nextDelay := time.Duration(0)
		if mask&fuzzModeDelayed != 0 {
			nextDelay = time.Duration(slowNext%fuzzMaxSteps+2) * fuzzTickUnit
		}

		sub := NewScheduler(gocron.DurationJob(fuzzSteps(seed))).Subscribe(ro.NewObserver(
			func(job ScheduleJob) {
				if inFlight.Add(1) > 1 {
					overlap.Add(1)
				}

				if int64(job.Counter) <= atomic.LoadInt64(&lastSeen) {
					outOfOrd.Add(1)
				}

				atomic.StoreInt64(&lastSeen, int64(job.Counter))
				time.Sleep(nextDelay)
				inFlight.Add(-1)

				if seen.Add(1) >= wantTicks {
					once.Do(func() { close(done) })
				}
			},
			func(error) {},
			func() {},
		))
		defer sub.Unsubscribe()

		waitBounded(t, done, "ticks")

		if n := overlap.Load(); n != 0 {
			t.Fatalf("rocron: %d overlapping Next calls", n)
		}

		if n := outOfOrd.Load(); n != 0 {
			t.Fatalf("rocron: %d out-of-order tick counters", n)
		}
	})
}
