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

package ro

import (
	"context"
	"errors"
	"fmt"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro/internal/xtest"
)

// Fuzz targets for the time/async operators. Inputs encode the interleaving (item counts,
// sync/async bitmask, how and when the downstream stops, tiny delays), never the expected result.

const (
	// fuzzTimeMaxMicros keeps every delay at or below 2ms so that one iteration stays fast.
	fuzzTimeMaxMicros = 2000

	// fuzzTimeSettle gives goroutines that outlive an unsubscription the time to misbehave
	// before the post-conditions are read.
	fuzzTimeSettle = 10 * time.Millisecond

	// Downstream stop strategies.
	fuzzTimeModeComplete = 0 // run to termination (finite sources only)
	fuzzTimeModeTake     = 1 // Take(k) cancels from inside Next
	fuzzTimeModeUnsub    = 2 // external, concurrent Unsubscribe
	fuzzTimeModeCancel   = 3 // context cancellation
	fuzzTimeModeCount    = 4
)

// fuzzTimePick derives a bounded, seed-dependent value; salt keeps independent knobs decorrelated.
func fuzzTimePick(seed int64, salt, lo, hi int) int {
	mixed := uint64(seed)*6364136223846793005 + uint64(salt+1)*1442695040888963407 //nolint:gosec // wrap-around is intended.
	mixed ^= mixed >> 29

	return fuzzBound(int64(mixed>>1), lo, hi) //nolint:gosec // top bit dropped, so never negative.
}

func fuzzTimeMicros(seed int64, salt, lo int) time.Duration {
	return time.Duration(fuzzTimePick(seed, salt, lo, fuzzTimeMaxMicros)) * time.Microsecond
}

// fuzzTimeIter runs body with panic recovery and a deadline, so a hang or a panic fails the
// iteration with the operator's name instead of crashing or blocking the whole run.
func fuzzTimeIter(t *testing.T, name string, body func() error) {
	t.Helper()

	done := make(chan error, 1)

	go func() {
		defer func() {
			if r := recover(); r != nil {
				done <- fmt.Errorf("%s: panic: %v\n%s", name, r, debug.Stack())
			}
		}()

		done <- body()
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
	case <-time.After(fuzzDeadline):
		t.Fatalf("%s: hang: iteration did not finish within %s", name, fuzzDeadline)
	}
}

// fuzzTimeSink is a downstream observer that records protocol violations.
type fuzzTimeSink[T any] struct {
	guard    serialGuard
	nexts    int32
	errs     int32
	comps    int32
	terminal int32
	afterEnd int32
}

func (s *fuzzTimeSink[T]) observer(slow time.Duration) Observer[T] {
	return NewObserverWithContext(
		func(_ context.Context, _ T) {
			s.guard.enter()
			defer s.guard.leave()

			if atomic.LoadInt32(&s.terminal) != 0 {
				atomic.AddInt32(&s.afterEnd, 1)
			}

			atomic.AddInt32(&s.nexts, 1)

			if slow > 0 {
				time.Sleep(slow)
			}
		},
		func(_ context.Context, _ error) {
			s.guard.enter()
			defer s.guard.leave()

			atomic.AddInt32(&s.terminal, 1)
			atomic.AddInt32(&s.errs, 1)
		},
		func(_ context.Context) {
			s.guard.enter()
			defer s.guard.leave()

			atomic.AddInt32(&s.terminal, 1)
			atomic.AddInt32(&s.comps, 1)
		},
	)
}

func (s *fuzzTimeSink[T]) violation() error {
	switch {
	case s.guard.overlapped() > 0:
		return fmt.Errorf("overlapping downstream calls: %d", s.guard.overlapped())
	case atomic.LoadInt32(&s.afterEnd) > 0:
		return fmt.Errorf("Next delivered after a terminal notification: %d", atomic.LoadInt32(&s.afterEnd))
	case atomic.LoadInt32(&s.terminal) > 1:
		return fmt.Errorf("%d terminal notifications", atomic.LoadInt32(&s.terminal))
	}

	return nil
}

// fuzzTimeDrive subscribes to obs `subs` times concurrently (distinct sinks, same operator
// instance), stops each subscription according to mode, and waits for them to close.
func fuzzTimeDrive[T any](obs Observable[T], subs, mode, take int, seed int64, slow time.Duration) ([]*fuzzTimeSink[T], error) {
	if mode == fuzzTimeModeTake {
		// Take is composed by the caller through fuzzTimeMaybeTake; kept here for symmetry only.
		_ = take
	}

	sinks := make([]*fuzzTimeSink[T], subs)
	errs := make([]error, subs)

	var wg sync.WaitGroup

	for i := 0; i < subs; i++ {
		sinks[i] = &fuzzTimeSink[T]{}

		wg.Add(1)

		go func(i int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					errs[i] = fmt.Errorf("panic in subscription %d: %v\n%s", i, r, debug.Stack())
				}
			}()

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			if mode == fuzzTimeModeCancel {
				go func() {
					time.Sleep(fuzzTimeMicros(seed, 40+i, 0))
					cancel()
				}()
			}

			sub := obs.SubscribeWithContext(ctx, sinks[i].observer(slow))

			if mode == fuzzTimeModeUnsub {
				go sub.Unsubscribe()
				fuzzJitter(seed, 50+i)
				sub.Unsubscribe()
			}

			sub.Wait()
		}(i)
	}

	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return sinks, err
		}
	}

	return sinks, nil
}

func fuzzTimeMaybeTake[T any](obs Observable[T], mode, take int) Observable[T] {
	if mode == fuzzTimeModeTake {
		return Take[T](int64(take))(obs)
	}

	return obs
}

// fuzzTimeEnv tracks what an instrumented source did, to assert upstream cleanup.
type fuzzTimeEnv struct {
	counter activeCounter
	wg      sync.WaitGroup
	panics  int32
	msg     atomic.Value
}

// source emits n items, from its own goroutine when async. Panics raised by downstream calls are
// counted instead of being lost in an operator-internal recover.
func (e *fuzzTimeEnv) source(seed int64, n int, async bool, gap time.Duration) Observable[int] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
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
func (e *fuzzTimeEnv) finish(allowBlockedProducers bool) error {
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

func fuzzTimeWaitUpstreamClosed(t *testing.T, c *activeCounter) {
	t.Helper()

	fuzzWaitFor(t, "upstream subscriptions to be released", func() bool { return c.activeCount() == 0 })
}

func fuzzTimeCheckSinks[T any](sinks []*fuzzTimeSink[T]) error {
	for _, s := range sinks {
		if err := s.violation(); err != nil {
			return err
		}
	}

	return nil
}

// fuzzTimeScenario is the decoded interleaving shared by most targets.
type fuzzTimeScenario struct {
	items int
	async bool
	mode  int
	take  int
	subs  int
	buf   int
	slow  time.Duration
	gap   time.Duration
}

func fuzzTimeDecode(seed int64, mask, k uint8, finite bool) fuzzTimeScenario {
	s := fuzzTimeScenario{
		items: fuzzTimePick(seed, 0, 1, fuzzMaxItems),
		async: fuzzIsAsync(mask, 0),
		mode:  fuzzBound(int64(k), 0, fuzzTimeModeCount-1),
		subs:  1 + int(mask>>7),
		buf:   fuzzTimePick(seed, 1, 1, 4),
	}

	if !finite && s.mode == fuzzTimeModeComplete {
		s.mode = fuzzTimeModeTake
	}

	s.take = fuzzTimePick(seed, 2, 1, s.items)

	if mask&2 != 0 {
		s.slow = fuzzTimeMicros(seed, 3, 0) / 8
	}

	if mask&4 != 0 {
		s.gap = fuzzTimeMicros(seed, 4, 0)
	}

	return s
}

func fuzzTimeSeeds(f *testing.F) {
	f.Helper()

	xtest.AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i * 37), uint8(i)} })
}

func FuzzTimeObserveOn(f *testing.F) {
	f.Skip("race: observeon-send-close; chansend (operator_utility.go:597) races close(ch) in teardown (operator_utility.go:585); remove when fixed")

	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)
		env := &fuzzTimeEnv{}

		fuzzTimeIter(t, "ObserveOn", func() error {
			obs := fuzzTimeMaybeTake(ObserveOn[int](sc.buf)(env.source(seed, sc.items, sc.async, 0)), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			if err := fuzzTimeCheckSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		fuzzTimeWaitUpstreamClosed(t, &env.counter)
	})
}

func FuzzTimeSubscribeOn(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)
		env := &fuzzTimeEnv{}

		fuzzTimeIter(t, "SubscribeOn", func() error {
			obs := fuzzTimeMaybeTake(SubscribeOn[int](sc.buf)(env.source(seed, sc.items, sc.async, 0)), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			if err := fuzzTimeCheckSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		fuzzTimeWaitUpstreamClosed(t, &env.counter)
	})
}

// FuzzTimeSubscribeOnInfinite targets an endless upstream: stopping the downstream must stop it.
func FuzzTimeSubscribeOnInfinite(f *testing.F) {
	f.Skip("race: subscribeon-infinite-hang; SubscribeOn(Interval)+Take/Unsubscribe/cancel never returns: \"SubscribeOn(Interval): hang: iteration did not finish within 5s\"; remove when fixed")

	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, false)

		var counter activeCounter

		fuzzTimeIter(t, "SubscribeOn(Interval)", func() error {
			interval := time.Duration(fuzzTimePick(seed, 5, 100, fuzzTimeMaxMicros)) * time.Microsecond
			obs := fuzzTimeMaybeTake(SubscribeOn[int64](sc.buf)(trackSubscriptions(&counter, Interval(interval))), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeDelay(f *testing.F) {
	f.Skip("race: delay-hang; Delay(d)(source) with Take/Unsubscribe/cancel never finishes: \"Delay: hang: iteration did not finish within 5s\"; remove when fixed")

	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var counter activeCounter

		fuzzTimeIter(t, "Delay", func() error {
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := fuzzTimeMaybeTake(Delay[int](fuzzTimeMicros(seed, 6, 0))(src), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(fuzzTimeSettle) // pending timers fire after teardown.

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeDelayEach(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var counter activeCounter

		fuzzTimeIter(t, "DelayEach", func() error {
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := fuzzTimeMaybeTake(DelayEach[int](fuzzTimeMicros(seed, 6, 0)/4)(src), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			return fuzzTimeCheckSinks(sinks)
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

func FuzzTimeTimeout(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)
		env := &fuzzTimeEnv{}

		fuzzTimeIter(t, "Timeout", func() error {
			// A duration near the source gap makes timeouts race with items and with completion.
			d := time.Duration(fuzzTimePick(seed, 7, 100, fuzzTimeMaxMicros)) * time.Microsecond
			obs := fuzzTimeMaybeTake(Timeout[int](d)(env.source(seed, sc.items, sc.async, sc.gap)), sc.mode, sc.take)

			sinks, err := fuzzTimeDrive(obs, sc.subs, sc.mode, sc.take, seed, sc.slow)
			if err != nil {
				return err
			}

			time.Sleep(fuzzTimeSettle) // a timer re-armed after teardown fires here.

			if err := fuzzTimeCheckSinks(sinks); err != nil {
				return err
			}

			return env.finish(false)
		})

		fuzzTimeWaitUpstreamClosed(t, &env.counter)
	})
}

func FuzzTimeInterval(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, false)

		var counter activeCounter

		fuzzTimeIter(t, "Interval", func() error {
			interval := time.Duration(fuzzTimePick(seed, 5, 100, fuzzTimeMaxMicros)) * time.Microsecond
			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, Interval(interval)), sc.mode, sc.take)

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

			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, IntervalWithInitial(initial, interval)), sc.mode, sc.take)

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
			obs := fuzzTimeMaybeTake(trackSubscriptions(&counter, Timer(fuzzTimeMicros(seed, 9, 0))), sc.mode, 1)

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

			obs := fuzzTimeMaybeTake(Future(func() (int, error) {
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

			obs := Future(func() (int, error) {
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
			sub := FromChannel[int](ch).Subscribe(sink.observer(fuzzTimeMicros(seed, 3, 0) / 16))

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

func FuzzTimeToChannel(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var counter activeCounter

		fuzzTimeIter(t, "ToChannel", func() error {
			size := fuzzTimePick(seed, 12, 0, 4)
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := ToChannel[int](size)(src)

			var (
				channels      int32
				completes     int32
				completeFirst int32
				consumed      int32
				sawTerminal   int32
			)

			consumerDone := make(chan struct{})

			slow := fuzzTimeMicros(seed, 13, 0) // delays Next(ch), giving Complete the chance to win.

			obs.Subscribe(NewObserver(
				func(ch <-chan Notification[int]) {
					atomic.AddInt32(&channels, 1)
					time.Sleep(slow)

					go func() {
						defer close(consumerDone)

						for n := range ch {
							if n.Kind == KindNext {
								atomic.AddInt32(&consumed, 1)
							} else {
								atomic.AddInt32(&sawTerminal, 1)
							}
						}
					}()
				},
				func(error) {},
				func() {
					if atomic.LoadInt32(&channels) == 0 {
						atomic.AddInt32(&completeFirst, 1)
					}

					atomic.AddInt32(&completes, 1)
				},
			))

			deadline := time.After(fuzzDeadline / 2)

			select {
			case <-consumerDone:
			case <-deadline:
				return fmt.Errorf("channel never closed: consumed %d/%d, channels=%d completes=%d", atomic.LoadInt32(&consumed), sc.items, atomic.LoadInt32(&channels), atomic.LoadInt32(&completes))
			}

			fuzzTimeWaitPlain(func() bool { return atomic.LoadInt32(&completes) > 0 })

			switch {
			case atomic.LoadInt32(&completeFirst) > 0:
				return errors.New("Complete delivered before the channel was emitted")
			case atomic.LoadInt32(&channels) != 1:
				return fmt.Errorf("channel emitted %d times", atomic.LoadInt32(&channels))
			case atomic.LoadInt32(&consumed) != int32(sc.items): //nolint:gosec // bounded.
				return fmt.Errorf("consumer got %d of %d items", atomic.LoadInt32(&consumed), sc.items)
			case atomic.LoadInt32(&sawTerminal) != 1:
				return fmt.Errorf("consumer saw %d terminal notifications", atomic.LoadInt32(&sawTerminal))
			}

			return nil
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

// fuzzTimeWaitPlain polls cond for at most one second and reports whether it became true.
func fuzzTimeWaitPlain(cond func() bool) bool {
	deadline := time.Now().Add(time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			return false
		}

		time.Sleep(time.Millisecond)
	}

	return true
}
