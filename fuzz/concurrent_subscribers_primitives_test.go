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
	"fmt"
	"runtime/debug"
	"sync"
	"testing"
	"time"

	"github.com/samber/ro"
)

const (
	// longestShortDelayMicros bounds the delays of time-based sources and downstreams, so that one
	// iteration stays fast while delays remain of the same order as the scheduling noise.
	longestShortDelayMicros = 2000

	// lateNotificationWait lets goroutines that outlive an unsubscription misbehave before the
	// post-conditions are read.
	lateNotificationWait = 10 * time.Millisecond

	// maxConcurrentSubscribers is how many goroutines subscribe to the same pipeline at once.
	maxConcurrentSubscribers = 2
)

// shortDelay maps a fuzz input to a delay between 0 and longestShortDelayMicros microseconds.
func shortDelay(micros uint16) time.Duration {
	return time.Duration(bounded(micros, 0, longestShortDelayMicros)) * time.Microsecond
}

// subscriberCount maps a fuzz input to the number of concurrent subscribers: one or two.
func subscriberCount(twoSubscribers bool) int {
	if twoSubscribers {
		return maxConcurrentSubscribers
	}

	return 1
}

// slowRecorder is a recorder whose callbacks take readDelay to run, like a slow downstream. A second
// overlap guard covers the whole callback, delay included, so that overlapping notifications are
// caught even when the recorder's own bookkeeping is too short to overlap.
type slowRecorder[T any] struct {
	*recorder[T]

	readDelay time.Duration
	readGuard overlapGuard
}

func newSlowRecorder[T any](readDelay time.Duration) *slowRecorder[T] {
	return &slowRecorder[T]{recorder: newRecorder[T](), readDelay: readDelay}
}

func (s *slowRecorder[T]) Next(value T) { s.NextWithContext(context.Background(), value) }

func (s *slowRecorder[T]) NextWithContext(ctx context.Context, value T) {
	s.readGuard.enter()
	defer s.readGuard.leave()

	time.Sleep(s.readDelay)
	s.recorder.NextWithContext(ctx, value)
}

func (s *slowRecorder[T]) Error(err error) { s.ErrorWithContext(context.Background(), err) }

func (s *slowRecorder[T]) ErrorWithContext(ctx context.Context, err error) {
	s.readGuard.enter()
	defer s.readGuard.leave()

	s.recorder.ErrorWithContext(ctx, err)
}

func (s *slowRecorder[T]) Complete() { s.CompleteWithContext(context.Background()) }

func (s *slowRecorder[T]) CompleteWithContext(ctx context.Context) {
	s.readGuard.enter()
	defer s.readGuard.leave()

	s.recorder.CompleteWithContext(ctx)
}

func (s *slowRecorder[T]) expectContract(t *testing.T) {
	t.Helper()

	s.readGuard.expectNone(t)
	s.recorder.expectContract(t)
}

// subscriptionEnding decides how one of several concurrent subscriptions ends. It gets the index of
// the subscriber, the cancel function of the subscriber context, and a function that subscribes.
// It returns the subscription, once the subscriber was told to stop in its own way.
type subscriptionEnding func(subscriber int, cancel context.CancelFunc, subscribe func() ro.Subscription) ro.Subscription

// endingWhenStreamEnds lets the stream end on its own: the source completes, or the pipeline
// stops itself, for example with ro.Take.
func endingWhenStreamEnds(_ int, _ context.CancelFunc, subscribe func() ro.Subscription) ro.Subscription {
	return subscribe()
}

// endingByUnsubscribe unsubscribes right after Subscribe returned, from this goroutine and from a
// second one at the same time, after yields chosen by yieldPattern.
func endingByUnsubscribe(yieldPattern uint8) subscriptionEnding {
	return func(subscriber int, _ context.CancelFunc, subscribe func() ro.Subscription) ro.Subscription {
		subscription := subscribe()

		go subscription.Unsubscribe()

		yieldAt(int64(yieldPattern), subscriber)
		subscription.Unsubscribe()

		return subscription
	}
}

// endingByContextCancel cancels the subscriber context from another goroutine after delay. The
// goroutine starts before Subscribe, so the cancellation can land at any point of the subscription.
func endingByContextCancel(delay time.Duration) subscriptionEnding {
	return func(_ int, cancel context.CancelFunc, subscribe func() ro.Subscription) ro.Subscription {
		go func() {
			time.Sleep(delay)
			cancel()
		}()

		return subscribe()
	}
}

// subscribeConcurrently subscribes `subscribers` slow recorders to pipeline, each from its own goroutine
// and with its own cancelable context. It ends every subscription as `ending` says, waits for it to
// finish, and returns the recorders. A hang or a panic in any subscriber fails the test.
func subscribeConcurrently[T any](t *testing.T, pipeline ro.Observable[T], subscribers int, readDelay time.Duration, ending subscriptionEnding) []*slowRecorder[T] {
	t.Helper()

	recorders := make([]*slowRecorder[T], subscribers)
	panics := make([]string, subscribers)

	runWithinDeadline(t, func() {
		var group sync.WaitGroup

		for subscriber := range recorders {
			recorders[subscriber] = newSlowRecorder[T](readDelay)

			group.Add(1)

			go func(subscriber int) {
				defer group.Done()
				defer func() {
					if recovered := recover(); recovered != nil {
						panics[subscriber] = fmt.Sprintf("panic in subscriber %d: %v\n%s", subscriber, recovered, debug.Stack())
					}
				}()

				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()

				subscription := ending(subscriber, cancel, func() ro.Subscription {
					return pipeline.SubscribeWithContext(ctx, recorders[subscriber])
				})
				subscription.Wait()
			}(subscriber)
		}

		group.Wait()

		for _, message := range panics {
			if message != "" {
				panic(message)
			}
		}
	})

	return recorders
}

// expectValidStreams waits for late notifications, then checks that every recorder saw a valid
// stream: no overlapping callbacks, at most one terminal notification, nothing after it.
func expectValidStreams[T any](t *testing.T, recorders []*slowRecorder[T]) {
	t.Helper()

	time.Sleep(lateNotificationWait)

	for _, recorder := range recorders {
		recorder.expectContract(t)
	}
}

// expectCleanShutdown is expectValidStreams plus a check that the upstream was released.
func expectCleanShutdown[T any](t *testing.T, recorders []*slowRecorder[T], upstream *subscriptionCounter) {
	t.Helper()

	expectValidStreams(t, recorders)
	upstream.expectAllReleased(t)
}
