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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSubscriberInternalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber1, ok1 := NewSubscriber(observer).(*subscriberImpl[int])
	subscriber2, ok2 := NewSafeSubscriber(observer).(*subscriberImpl[int])
	subscriber3, ok3 := NewUnsafeSubscriber(observer).(*subscriberImpl[int])
	subscriber4, ok4 := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])
	subscriber5, ok5 := NewSingleProducerSubscriber(observer).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)
	is.True(ok5)

	// default state
	is.EqualValues(KindNext, subscriber1.status)
	is.EqualValues(KindNext, subscriber2.status)
	is.EqualValues(KindNext, subscriber3.status)
	is.EqualValues(KindNext, subscriber4.status)
	is.EqualValues(KindNext, subscriber5.status)

	// send values
	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(KindNext, subscriber1.status)
	is.EqualValues(KindNext, subscriber2.status)
	is.EqualValues(KindNext, subscriber3.status)
	is.EqualValues(KindNext, subscriber4.status)
	is.EqualValues(KindNext, subscriber5.status)

	// completed state
	subscriber1.Complete()
	subscriber2.Complete()
	subscriber3.Complete()
	subscriber4.Complete()
	subscriber5.Complete()
	is.EqualValues(KindComplete, subscriber1.status)
	is.EqualValues(KindComplete, subscriber2.status)
	is.EqualValues(KindComplete, subscriber3.status)
	is.EqualValues(KindComplete, subscriber4.status)
	is.EqualValues(KindComplete, subscriber5.status)

	// no change
	subscriber1.Next(42)
	subscriber2.Next(42)
	subscriber3.Next(42)
	subscriber4.Next(42)
	subscriber5.Next(42)
	is.EqualValues(KindComplete, subscriber1.status)
	is.EqualValues(KindComplete, subscriber2.status)
	is.EqualValues(KindComplete, subscriber3.status)
	is.EqualValues(KindComplete, subscriber4.status)
	is.EqualValues(KindComplete, subscriber5.status)
}

func TestSubscriberInternalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber1, ok1 := NewSubscriber(observer).(*subscriberImpl[int])
	subscriber2, ok2 := NewSafeSubscriber(observer).(*subscriberImpl[int])
	subscriber3, ok3 := NewUnsafeSubscriber(observer).(*subscriberImpl[int])
	subscriber4, ok4 := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])
	subscriber5, ok5 := NewSingleProducerSubscriber(observer).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)
	is.True(ok5)

	// default state
	is.EqualValues(KindNext, subscriber1.status)
	is.EqualValues(KindNext, subscriber2.status)
	is.EqualValues(KindNext, subscriber3.status)
	is.EqualValues(KindNext, subscriber4.status)

	// send values
	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(KindNext, subscriber1.status)
	is.EqualValues(KindNext, subscriber2.status)
	is.EqualValues(KindNext, subscriber3.status)
	is.EqualValues(KindNext, subscriber4.status)
	is.EqualValues(KindNext, subscriber5.status)

	// trigger error
	subscriber1.Error(assert.AnError)
	subscriber2.Error(assert.AnError)
	subscriber3.Error(assert.AnError)
	subscriber4.Error(assert.AnError)
	subscriber5.Error(assert.AnError)
	is.EqualValues(KindError, subscriber1.status)
	is.EqualValues(KindError, subscriber2.status)
	is.EqualValues(KindError, subscriber3.status)
	is.EqualValues(KindError, subscriber4.status)
	is.EqualValues(KindError, subscriber5.status)

	// no change
	subscriber1.Next(42)
	subscriber2.Next(42)
	subscriber3.Next(42)
	subscriber4.Next(42)
	subscriber5.Next(42)
	is.EqualValues(KindError, subscriber1.status)
	is.EqualValues(KindError, subscriber2.status)
	is.EqualValues(KindError, subscriber3.status)
	is.EqualValues(KindError, subscriber4.status)
	is.EqualValues(KindError, subscriber5.status)
}

func TestSubscriberNext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	var counter1 int64
	var counter2 int64
	var counter3 int64
	var counter4 int64
	var counter5 int64

	observer1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) {},
		func() {},
	)
	observer2 := NewObserver(
		func(value int) { atomic.AddInt64(&counter2, int64(value)) },
		func(err error) {},
		func() {},
	)
	observer3 := NewObserver(
		func(value int) { atomic.AddInt64(&counter3, int64(value)) },
		func(err error) {},
		func() {},
	)
	observer4 := NewObserver(
		func(value int) { atomic.AddInt64(&counter4, int64(value)) },
		func(err error) {},
		func() {},
	)
	observer5 := NewObserver(
		func(value int) { atomic.AddInt64(&counter5, int64(value)) },
		func(err error) {},
		func() {},
	)

	subscriber1, ok1 := NewSubscriber(observer1).(*subscriberImpl[int])
	subscriber2, ok2 := NewSafeSubscriber(observer2).(*subscriberImpl[int])
	subscriber3, ok3 := NewUnsafeSubscriber(observer3).(*subscriberImpl[int])
	subscriber4, ok4 := NewEventuallySafeSubscriber(observer4).(*subscriberImpl[int])
	subscriber5, ok5 := NewSingleProducerSubscriber(observer5).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)
	is.True(ok5)

	subscriber1.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(0, atomic.LoadInt64(&counter2))
	is.EqualValues(0, atomic.LoadInt64(&counter3))
	is.EqualValues(0, atomic.LoadInt64(&counter4))
	is.EqualValues(0, atomic.LoadInt64(&counter5))

	subscriber2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(0, atomic.LoadInt64(&counter3))
	is.EqualValues(0, atomic.LoadInt64(&counter4))
	is.EqualValues(0, atomic.LoadInt64(&counter5))

	subscriber3.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(0, atomic.LoadInt64(&counter4))
	is.EqualValues(0, atomic.LoadInt64(&counter5))

	subscriber4.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(0, atomic.LoadInt64(&counter5))

	subscriber5.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	subscriber1.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	subscriber2.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(42, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	subscriber3.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(42, atomic.LoadInt64(&counter2))
	is.EqualValues(42, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	subscriber4.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(42, atomic.LoadInt64(&counter2))
	is.EqualValues(42, atomic.LoadInt64(&counter3))
	is.EqualValues(42, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	subscriber5.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(42, atomic.LoadInt64(&counter2))
	is.EqualValues(42, atomic.LoadInt64(&counter3))
	is.EqualValues(42, atomic.LoadInt64(&counter4))
	is.EqualValues(42, atomic.LoadInt64(&counter5))
}

func TestSubscriberError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter1 int64
	var counter2 int64
	var counter3 int64
	var counter4 int64
	var counter5 int64

	observer1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) { atomic.AddInt64(&counter1, int64(1)) },
		func() {},
	)
	observer2 := NewObserver(
		func(value int) { atomic.AddInt64(&counter2, int64(value)) },
		func(err error) { atomic.AddInt64(&counter2, int64(1)) },
		func() {},
	)
	observer3 := NewObserver(
		func(value int) { atomic.AddInt64(&counter3, int64(value)) },
		func(err error) { atomic.AddInt64(&counter3, int64(1)) },
		func() {},
	)
	observer4 := NewObserver(
		func(value int) { atomic.AddInt64(&counter4, int64(value)) },
		func(err error) { atomic.AddInt64(&counter4, int64(1)) },
		func() {},
	)
	observer5 := NewObserver(
		func(value int) { atomic.AddInt64(&counter5, int64(value)) },
		func(err error) { atomic.AddInt64(&counter5, int64(1)) },
		func() {},
	)

	subscriber1, ok1 := NewSubscriber(observer1).(*subscriberImpl[int])
	subscriber2, ok2 := NewSafeSubscriber(observer2).(*subscriberImpl[int])
	subscriber3, ok3 := NewUnsafeSubscriber(observer3).(*subscriberImpl[int])
	subscriber4, ok4 := NewEventuallySafeSubscriber(observer4).(*subscriberImpl[int])
	subscriber5, ok5 := NewSingleProducerSubscriber(observer5).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)
	is.True(ok5)

	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	// trigger error
	subscriber1.Error(assert.AnError)
	subscriber2.Error(assert.AnError)
	subscriber3.Error(assert.AnError)
	subscriber4.Error(assert.AnError)
	subscriber5.Error(assert.AnError)
	is.EqualValues(22, atomic.LoadInt64(&counter1))
	is.EqualValues(22, atomic.LoadInt64(&counter2))
	is.EqualValues(22, atomic.LoadInt64(&counter3))
	is.EqualValues(22, atomic.LoadInt64(&counter4))
	is.EqualValues(22, atomic.LoadInt64(&counter5))

	// send a new message
	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(22, atomic.LoadInt64(&counter1))
	is.EqualValues(22, atomic.LoadInt64(&counter2))
	is.EqualValues(22, atomic.LoadInt64(&counter3))
	is.EqualValues(22, atomic.LoadInt64(&counter4))
	is.EqualValues(22, atomic.LoadInt64(&counter5))
}

func TestSubscriberComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter1 int64
	var counter2 int64
	var counter3 int64
	var counter4 int64
	var counter5 int64

	observer1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&counter1, 1) },
	)
	observer2 := NewObserver(
		func(value int) { atomic.AddInt64(&counter2, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&counter2, 1) },
	)
	observer3 := NewObserver(
		func(value int) { atomic.AddInt64(&counter3, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&counter3, 1) },
	)
	observer4 := NewObserver(
		func(value int) { atomic.AddInt64(&counter4, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&counter4, 1) },
	)
	observer5 := NewObserver(
		func(value int) { atomic.AddInt64(&counter5, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&counter5, 1) },
	)

	subscriber1, ok1 := NewSubscriber(observer1).(*subscriberImpl[int])
	subscriber2, ok2 := NewSafeSubscriber(observer2).(*subscriberImpl[int])
	subscriber3, ok3 := NewUnsafeSubscriber(observer3).(*subscriberImpl[int])
	subscriber4, ok4 := NewEventuallySafeSubscriber(observer4).(*subscriberImpl[int])
	subscriber5, ok5 := NewSingleProducerSubscriber(observer5).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)
	is.True(ok5)

	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
	is.EqualValues(21, atomic.LoadInt64(&counter3))
	is.EqualValues(21, atomic.LoadInt64(&counter4))
	is.EqualValues(21, atomic.LoadInt64(&counter5))

	// trigger complete
	subscriber1.Complete()
	subscriber2.Complete()
	subscriber3.Complete()
	subscriber4.Complete()
	subscriber5.Complete()
	is.EqualValues(22, atomic.LoadInt64(&counter1))
	is.EqualValues(22, atomic.LoadInt64(&counter2))
	is.EqualValues(22, atomic.LoadInt64(&counter3))
	is.EqualValues(22, atomic.LoadInt64(&counter4))
	is.EqualValues(22, atomic.LoadInt64(&counter5))

	// send a new message
	subscriber1.Next(21)
	subscriber2.Next(21)
	subscriber3.Next(21)
	subscriber4.Next(21)
	subscriber5.Next(21)
	is.EqualValues(22, atomic.LoadInt64(&counter1))
	is.EqualValues(22, atomic.LoadInt64(&counter2))
	is.EqualValues(22, atomic.LoadInt64(&counter3))
	is.EqualValues(22, atomic.LoadInt64(&counter4))
	is.EqualValues(22, atomic.LoadInt64(&counter5))
}

func TestSubscriberWithContext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	type contextKey string
	const contextKeyTest = contextKey("test")

	var receivedValue int
	var receivedError error
	var completed bool

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			receivedValue = value
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
		func(ctx context.Context, err error) {
			receivedError = err

			is.Fail("should not be called")
		},
		func(ctx context.Context) {
			completed = true
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	ctx := context.WithValue(context.Background(), contextKeyTest, "value")

	// Test NextWithContext
	subscriber.NextWithContext(ctx, 42)
	is.Equal(42, receivedValue)

	// Test CompleteWithContext
	subscriber.CompleteWithContext(ctx)
	is.True(completed)

	// Create new subscriber for error test
	observer2 := NewObserverWithContext(
		func(ctx context.Context, value int) {
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
		func(ctx context.Context, err error) {
			receivedError = err
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
		func(ctx context.Context) {
			is.Fail("should not be called")
		},
	)

	subscriber2, ok2 := NewSubscriber(observer2).(*subscriberImpl[int])
	is.True(ok2)

	// Test ErrorWithContext
	subscriber2.ErrorWithContext(ctx, assert.AnError)
	is.Equal(assert.AnError, receivedError)
}

func TestLocklessDroppedNotification(t *testing.T) {
	// Do NOT run this test in parallel. It mutates the global
	// `OnDroppedNotification` hook which may race with other tests
	// that read or call the hook. Keep the existing defer that
	// restores the previous handler so the hook is reset for later
	// tests.
	is := assert.New(t)

	// Capture dropped notifications using the helper which serializes
	// overrides of the global `OnDroppedNotification` hook and restores it
	// after the inner function returns.
	var seen string
	WithDroppedNotification(t, func(ctx context.Context, notification fmt.Stringer) {
		seen = notification.String()
	}, func() {
		observer := NewObserver(
			func(value int) {},
			func(err error) {},
			func() {},
		)

		subscriber, ok := NewSingleProducerSubscriber(observer).(*subscriberImpl[int])
		is.True(ok)

		// Mark as completed so further Next() should be dropped
		subscriber.Complete()

		subscriber.Next(42)

		is.NotEmpty(seen)
		is.Contains(seen, "Next(42)")
	})
}

// Stress test for the single-producer fast path. This exercises the lockless
// code path under a tight sequential loop to ensure correctness and to make it
// easy to run under the race detector in CI.
func TestSingleProducerStress(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter int64
	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSingleProducerSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	const iterations = 100000
	for i := 0; i < iterations; i++ {
		subscriber.Next(1)
	}

	is.Equal(int64(iterations), atomic.LoadInt64(&counter))
}

// TestSingleProducerContextCancellation ensures the single-producer (lockless)
// fast path forwards the provided context to the destination observer and that
// a cancelled context is visible to the observer callback.
func TestSingleProducerContextCancellation(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	var sawCanceled bool

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			if ctx.Err() == context.Canceled {
				sawCanceled = true
			}
		},
		func(ctx context.Context, err error) {},
		func(ctx context.Context) {},
	)

	subscriber, ok := NewSingleProducerSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	subscriber.NextWithContext(ctx, 42)

	is.True(sawCanceled)
}

// Concurrent stress test for the safe subscriber. Spawns multiple goroutines
// concurrently calling Next and validates the final accumulated value.
func TestSafeSubscriberConcurrentStress(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter int64
	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	const goroutines = 8
	const per = 10000

	var wg sync.WaitGroup
	wg.Add(goroutines)

	for g := 0; g < goroutines; g++ {
		go func() {
			for i := 0; i < per; i++ {
				subscriber.Next(1)
			}
			wg.Done()
		}()
	}

	wg.Wait()

	expected := int64(goroutines * per)
	is.Equal(expected, atomic.LoadInt64(&counter))
}

func TestSubscriberIsClosed(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	// Initially not closed
	is.False(subscriber.IsClosed())

	// After error, should be closed
	subscriber.Error(assert.AnError)
	is.True(subscriber.IsClosed())

	// Create new subscriber for complete test
	observer2 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)
	subscriber2, ok2 := NewSubscriber(observer2).(*subscriberImpl[int])

	is.True(ok2)

	// After complete, should be closed
	subscriber2.Complete()
	is.True(subscriber2.IsClosed())
}

func TestSubscriberHasThrown(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	// Initially not thrown
	is.False(subscriber.HasThrown())

	// After error, should be thrown
	subscriber.Error(assert.AnError)
	is.True(subscriber.HasThrown())

	// Create new subscriber for complete test
	observer2 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)
	subscriber2, ok2 := NewSubscriber(observer2).(*subscriberImpl[int])

	is.True(ok2)

	// After complete, should not be thrown
	subscriber2.Complete()
	is.False(subscriber2.HasThrown())
}

func TestSubscriberIsCompleted(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])
	is.True(ok)

	// Initially not completed
	is.False(subscriber.IsCompleted())

	// After complete, should be completed
	subscriber.Complete()
	is.True(subscriber.IsCompleted())

	// Create new subscriber for error test
	observer2 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)
	subscriber2, ok2 := NewSubscriber(observer2).(*subscriberImpl[int])

	is.True(ok2)

	// After error, should not be completed
	subscriber2.Error(assert.AnError)
	is.False(subscriber2.IsCompleted())
}

func TestSubscriberUnsubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var teardownCalled bool

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	subscriber.Add(func() {
		teardownCalled = true
	})

	// Initially not closed
	is.False(subscriber.IsClosed())
	is.False(teardownCalled)

	// After unsubscribe, should be closed
	subscriber.Unsubscribe()
	is.True(subscriber.IsClosed())
	is.True(teardownCalled)

	// Multiple unsubscribe calls should be safe
	teardownCalled = false

	subscriber.Unsubscribe()
	is.True(subscriber.IsClosed())
	is.False(teardownCalled) // Should not call teardown again
}

func TestSubscriberEventuallySafeBackpressureDrop(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test that backpressure is set correctly
	is.Equal(BackpressureDrop, subscriber.backpressure)

	// Send values normally
	subscriber.Next(21)
	subscriber.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter))
}

func TestSubscriberEventuallySafeBackpressureDropConcurrent(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) {
			time.Sleep(100 * time.Millisecond)
			atomic.AddInt64(&counter, int64(value))
		},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test that backpressure is set correctly
	is.Equal(BackpressureDrop, subscriber.backpressure)

	// One goroutine sends a value, the other one should be dropped
	go subscriber.Next(21)
	go subscriber.Next(21)

	is.Equal(int64(0), atomic.LoadInt64(&counter))

	time.Sleep(150 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter))

	time.Sleep(100 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter))
}

func TestSubscriberWrappingExistingSubscriber(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	// Create a subscriber
	subscriber1 := NewSubscriber(observer)

	// Wrap it in another subscriber - should return the same instance
	subscriber2 := NewSubscriber[int](subscriber1)

	// They should be the same instance
	is.Equal(subscriber1, subscriber2)

	// Test with different constructor functions
	subscriber3 := NewSafeSubscriber[int](subscriber1)
	subscriber4 := NewUnsafeSubscriber[int](subscriber1)
	subscriber5 := NewEventuallySafeSubscriber[int](subscriber1)

	is.Equal(subscriber1, subscriber3)
	is.Equal(subscriber1, subscriber4)
	is.Equal(subscriber1, subscriber5)
}

func TestSubscriberWithSubscriptionObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var teardownCalled bool

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	// Create a subscription
	subscription := NewSubscription(func() {
		teardownCalled = true
	})

	// Create a subscriber that wraps both observer and subscription
	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Add the subscription to the subscriber
	subscriber.AddUnsubscribable(subscription)

	// Initially not called
	is.False(teardownCalled)

	// After unsubscribe, teardown should be called
	subscriber.Unsubscribe()
	is.True(teardownCalled)
}

func TestSubscriberConcurrentAccess(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent Next calls
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				subscriber.Next(j)
			}
		}()
	}

	wg.Wait()
	is.EqualValues(450, atomic.LoadInt64(&counter)) // 10 * sum(0..9) = 450
}

// This test should be executed with -race flag, because it tests concurrent access to the subscriber.
// It is not a problem for the safe subscriber, but it is for the eventually safe subscriber.
func TestSubscriberEventuallySafeConcurrentAccess(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) {
			// Very short sleep to increase chance of race condition.
			// On concurrent access, the value will be dropped.
			time.Sleep(100 * time.Microsecond)
			atomic.AddInt64(&counter, int64(value))
		},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent Next calls
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				subscriber.Next(j)
			}
		}()
	}

	wg.Wait()

	// The counter should not be 4500 because some values have been dropped.
	is.NotEqualValues(4500, atomic.LoadInt64(&counter))
}

func TestSubscriberNilObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with nil observer - should not panic
	subscriber := NewSubscriber[int](nil)
	is.NotNil(subscriber)

	// The subscriber should be created but calling methods will cause panics
	// because the destination is nil. This is expected behavior.
	// We test that the subscriber is created successfully.
	is.False(subscriber.IsClosed())

	subscriber.Next(21)

	subscriber.Unsubscribe()
	is.True(subscriber.IsClosed())
	is.True(subscriber.IsCompleted())
	is.False(subscriber.HasThrown())
}

func TestSubscriberStatusTransitions(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Initial state
	is.EqualValues(KindNext, subscriber.status)
	is.False(subscriber.IsClosed())
	is.False(subscriber.HasThrown())
	is.False(subscriber.IsCompleted())

	// After error
	subscriber.Error(assert.AnError)
	is.EqualValues(KindError, subscriber.status)
	is.True(subscriber.IsClosed())
	is.True(subscriber.HasThrown())
	is.False(subscriber.IsCompleted())

	// Create new subscriber for complete test
	observer2 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)
	subscriber2, ok2 := NewSubscriber(observer2).(*subscriberImpl[int])

	is.True(ok2)

	// After complete
	subscriber2.Complete()
	is.EqualValues(KindComplete, subscriber2.status)
	is.True(subscriber2.IsClosed())
	is.False(subscriber2.HasThrown())
	is.True(subscriber2.IsCompleted())

	// Create new subscriber for unsubscribe test
	observer3 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)
	subscriber3, ok3 := NewSubscriber(observer3).(*subscriberImpl[int])

	is.True(ok3)

	// After unsubscribe
	subscriber3.Unsubscribe()
	is.EqualValues(KindComplete, subscriber3.status)
	is.True(subscriber3.IsClosed())
	is.False(subscriber3.HasThrown())
	is.True(subscriber3.IsCompleted())
}

func TestSubscriberConcurrentMixedOperations(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	var nextCounter int64
	var errorCounter int64
	var completeCounter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&nextCounter, int64(value)) },
		func(err error) { atomic.AddInt64(&errorCounter, 1) },
		func() { atomic.AddInt64(&completeCounter, 1) },
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent Next, Error, Complete, and Unsubscribe calls
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				switch j % 4 {
				case 0:
					subscriber.Next(j)
				case 1:
					subscriber.Error(assert.AnError)
				case 2:
					subscriber.Complete()
				case 3:
					subscriber.Unsubscribe()
				}
			}
		}()
	}

	wg.Wait()

	// Verify that the subscriber is in a consistent state
	is.True(subscriber.IsClosed())

	// At least one operation should have been processed
	is.Equal(int64(1), atomic.LoadInt64(&errorCounter)+atomic.LoadInt64(&completeCounter))
}

func TestSubscriberConcurrentStatusChecks(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent status checks
	var wg sync.WaitGroup

	var isClosedCount int64

	var hasThrownCount int64

	var isCompletedCount int64

	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 100; j++ {
				if subscriber.IsClosed() {
					atomic.AddInt64(&isClosedCount, 1)
				}

				if subscriber.HasThrown() {
					atomic.AddInt64(&hasThrownCount, 1)
				}

				if subscriber.IsCompleted() {
					atomic.AddInt64(&isCompletedCount, 1)
				}
			}
		}()
	}

	wg.Wait()

	// All status checks should be consistent
	is.EqualValues(0, atomic.LoadInt64(&isClosedCount))
	is.EqualValues(0, atomic.LoadInt64(&hasThrownCount))
	is.EqualValues(0, atomic.LoadInt64(&isCompletedCount))
}

func TestSubscriberConcurrentContextOperations(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 300*time.Millisecond)
	is := assert.New(t)

	type contextKey string

	const contextKeyGoroutine = contextKey("goroutine")

	var nextCounter int64

	var errorCounter int64

	var completeCounter int64

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			v, ok := ctx.Value(contextKeyGoroutine).(int)
			is.True(ok)
			is.Equal(42, v)
			atomic.AddInt64(&nextCounter, int64(value))
		},
		func(ctx context.Context, err error) {
			v, ok := ctx.Value(contextKeyGoroutine).(int)
			is.True(ok)
			is.Equal(42, v)
			atomic.AddInt64(&errorCounter, 1)
		},
		func(ctx context.Context) {
			v, ok := ctx.Value(contextKeyGoroutine).(int)
			is.True(ok)
			is.Equal(42, v)
			atomic.AddInt64(&completeCounter, 1)
		},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent context operations
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			ctx := context.WithValue(context.Background(), contextKeyGoroutine, 42)

			for j := 0; j < 10; j++ {
				switch j % 3 {
				case 0:
					subscriber.NextWithContext(ctx, j)
				case 1:
					subscriber.ErrorWithContext(ctx, assert.AnError)
				case 2:
					subscriber.CompleteWithContext(ctx)
				}
			}
		}()
	}

	wg.Wait()

	// Verify that the subscriber is in a consistent state
	is.True(subscriber.IsClosed())
}

func TestSubscriberConcurrentUnsubscribe(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var teardownCounter int64

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	subscriber.Add(func() {
		atomic.AddInt64(&teardownCounter, 1)
	})

	// Test concurrent unsubscribe calls
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			subscriber.Unsubscribe()
		}()
	}

	wg.Wait()

	// Teardown should be called exactly once
	is.EqualValues(1, atomic.LoadInt64(&teardownCounter))
	is.True(subscriber.IsClosed())
}

func TestSubscriberConcurrentAddUnsubscribable(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var teardownCounter int64

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent AddUnsubscribable calls
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			subscription := NewSubscription(func() {
				atomic.AddInt64(&teardownCounter, 1)
			})
			subscriber.AddUnsubscribable(subscription)
		}()
	}

	wg.Wait()

	// Unsubscribe should trigger all teardown functions
	subscriber.Unsubscribe()
	is.EqualValues(50, atomic.LoadInt64(&teardownCounter))
}

func TestSubscriberConcurrentSafeVsUnsafe(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 300*time.Millisecond)
	is := assert.New(t)

	var safeCounter int64

	var unsafeCounter int64

	safeObserver := NewObserver(
		func(value int) { atomic.AddInt64(&safeCounter, int64(value)) },
		func(err error) {},
		func() {},
	)

	unsafeObserver := NewObserver(
		func(value int) { atomic.AddInt64(&unsafeCounter, int64(value)) },
		func(err error) {},
		func() {},
	)

	safeSubscriber, ok1 := NewSafeSubscriber(safeObserver).(*subscriberImpl[int])
	unsafeSubscriber, ok2 := NewUnsafeSubscriber(unsafeObserver).(*subscriberImpl[int])

	is.True(ok1)
	is.True(ok2)

	// Test concurrent access - safe should handle it properly, unsafe may have race conditions
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				safeSubscriber.Next(j)
				unsafeSubscriber.Next(j)
			}
		}()
	}

	wg.Wait()

	// Safe subscriber should have consistent results
	is.EqualValues(4500, atomic.LoadInt64(&safeCounter)) // 100 * sum(0..9) = 4500

	// Unsafe subscriber may have race conditions, so we just check it's not zero
	is.Positive(atomic.LoadInt64(&unsafeCounter))
}

func TestSubscriberConcurrentEventuallySafeDropBehavior(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 300*time.Millisecond)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) {
			// Simulate slow processing to increase chance of drops
			time.Sleep(10 * time.Millisecond)
			atomic.AddInt64(&counter, int64(value))
		},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewEventuallySafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test that rapid concurrent calls result in dropped messages
	var wg sync.WaitGroup
	for i := 0; i < 200; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				subscriber.Next(j)
			}
		}()
	}

	wg.Wait()

	// The counter should be less than expected due to drops
	expected := int64(200 * 45) // 200 * sum(0..9)
	is.Less(atomic.LoadInt64(&counter), expected)
	is.Positive(atomic.LoadInt64(&counter)) // But some should get through
}

func TestSubscriberConcurrentErrorHandling(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var errorCounter int64
	var nextCounter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&nextCounter, int64(value)) },
		func(err error) { atomic.AddInt64(&errorCounter, 1) },
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent error and next calls
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func(j int) {
			defer wg.Done()

			if j%2 == 0 {
				subscriber.Error(assert.AnError)
			} else {
				subscriber.Next(j)
			}
		}(i)
	}

	wg.Wait()

	// Should be in error state
	is.True(subscriber.HasThrown())
	is.True(subscriber.IsClosed())
	is.False(subscriber.IsCompleted())

	// At least one error should be processed
	is.GreaterOrEqual(int64(1), atomic.LoadInt64(&errorCounter))
}

func TestSubscriberConcurrentCompleteHandling(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	var completeCounter int64
	var nextCounter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&nextCounter, int64(value)) },
		func(err error) {},
		func() { atomic.AddInt64(&completeCounter, 1) },
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent complete and next calls
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func(j int) {
			defer wg.Done()

			if j%2 == 0 {
				subscriber.Complete()
			} else {
				subscriber.Next(j)
			}
		}(i)
	}

	wg.Wait()

	// Should be in complete state
	is.False(subscriber.HasThrown())
	is.True(subscriber.IsClosed())
	is.True(subscriber.IsCompleted())

	// At least one complete should be processed
	is.GreaterOrEqual(int64(1), atomic.LoadInt64(&completeCounter))
}

func TestSubscriberConcurrentNilObserver(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subscriber, ok := NewSafeSubscriber[int](nil).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent operations with nil observer
	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				subscriber.Next(j)
				subscriber.Error(assert.AnError)
				subscriber.Complete()
				subscriber.Unsubscribe()
			}
		}()
	}

	wg.Wait()

	// Should be in a consistent state
	is.True(subscriber.IsClosed())
}

func TestSubscriberConcurrentStatusTransitions(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 300*time.Millisecond)
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	subscriber, ok := NewSafeSubscriber(observer).(*subscriberImpl[int])

	is.True(ok)

	// Test concurrent status transitions
	var wg sync.WaitGroup
	var finalStatus int32

	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func(i int) {
			defer wg.Done()

			switch i % 3 {
			case 0:
				subscriber.Error(assert.AnError)
			case 1:
				subscriber.Complete()
			case 2:
				subscriber.Unsubscribe()
			}
		}(i)
	}

	wg.Wait()

	// Check final status
	finalStatus = subscriber.status
	is.True(finalStatus == 1 || finalStatus == 2)
	is.True(subscriber.IsClosed())
}
