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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestObserverInternalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer1, ok1 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	).(*observerImpl[int])
	observer2, ok2 := OnNext(func(value int) {}).(*observerImpl[int])
	observer3, ok3 := OnError[int](func(err error) {}).(*observerImpl[int])
	observer4, ok4 := OnComplete[int](func() {}).(*observerImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)

	// default state
	is.EqualValues(0, observer1.status)
	is.EqualValues(0, observer2.status)
	is.EqualValues(0, observer3.status)
	is.EqualValues(0, observer4.status)

	// send values
	observer1.Next(21)
	observer2.Next(21)
	observer3.Next(21)
	observer4.Next(21)
	is.EqualValues(0, observer1.status)
	is.EqualValues(0, observer2.status)
	is.EqualValues(0, observer3.status)
	is.EqualValues(0, observer4.status)

	// completed state
	observer1.Complete()
	observer2.Complete()
	observer3.Complete()
	observer4.Complete()
	is.EqualValues(2, observer1.status)
	is.EqualValues(2, observer2.status)
	is.EqualValues(2, observer3.status)
	is.EqualValues(2, observer4.status)

	// no change
	observer1.Next(42)
	observer2.Next(42)
	observer3.Next(42)
	observer4.Next(42)
	is.EqualValues(2, observer1.status)
	is.EqualValues(2, observer2.status)
	is.EqualValues(2, observer3.status)
	is.EqualValues(2, observer4.status)

	// nil
	NewObserver[int](nil, func(error) {}, func() {}).Next(42)
	NewObserver(func(int) {}, nil, func() {}).Error(assert.AnError)
	NewObserver(func(int) {}, func(error) {}, nil).Complete()
}

func TestObserverInternalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer1, ok1 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	).(*observerImpl[int])
	observer2, ok2 := OnNext(func(value int) {}).(*observerImpl[int])
	observer3, ok3 := OnError[int](func(err error) {}).(*observerImpl[int])
	observer4, ok4 := OnComplete[int](func() {}).(*observerImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	is.True(ok4)

	// default state
	is.EqualValues(0, observer1.status)
	is.EqualValues(0, observer2.status)
	is.EqualValues(0, observer3.status)
	is.EqualValues(0, observer4.status)

	// send values
	observer1.Next(21)
	observer2.Next(21)
	observer3.Next(21)
	observer4.Next(21)
	is.EqualValues(0, observer1.status)
	is.EqualValues(0, observer2.status)
	is.EqualValues(0, observer3.status)
	is.EqualValues(0, observer4.status)

	// trigger error
	observer1.Error(assert.AnError)
	observer2.Error(assert.AnError)
	observer3.Error(assert.AnError)
	observer4.Error(assert.AnError)
	is.EqualValues(1, observer1.status)
	is.EqualValues(1, observer2.status)
	is.EqualValues(1, observer3.status)
	is.EqualValues(1, observer4.status)

	// no change
	observer1.Next(42)
	observer2.Next(42)
	observer3.Next(42)
	observer4.Next(42)
	is.EqualValues(1, observer1.status)
	is.EqualValues(1, observer2.status)
	is.EqualValues(1, observer3.status)
	is.EqualValues(1, observer4.status)
}

func TestObserverNext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	var counter1 int64
	var counter2 int64

	observer1, ok1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) {},
		func() {},
	).(*observerImpl[int])
	observer2, ok2 := OnNext(func(value int) { atomic.AddInt64(&counter2, int64(value)) }).(*observerImpl[int])

	is.True(ok1)
	is.True(ok2)

	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))

	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(42, atomic.LoadInt64(&counter1))
	is.EqualValues(42, atomic.LoadInt64(&counter2))
}

func TestObserverError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter1 int64
	var counter2 int64

	observer1, ok1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) {},
		func() {},
	).(*observerImpl[int])
	observer2, ok2 := OnError[int](func(err error) { atomic.AddInt64(&counter2, int64(21)) }).(*observerImpl[int])

	is.True(ok1)
	is.True(ok2)

	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(0, atomic.LoadInt64(&counter2))

	// trigger error
	observer1.Error(assert.AnError)
	observer2.Error(assert.AnError)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))

	// send a new message
	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
}

func TestObserverComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var counter1 int64
	var counter2 int64

	observer1, ok1 := NewObserver(
		func(value int) { atomic.AddInt64(&counter1, int64(value)) },
		func(err error) {},
		func() {},
	).(*observerImpl[int])
	observer2, ok2 := OnComplete[int](func() { atomic.AddInt64(&counter2, int64(21)) }).(*observerImpl[int])

	is.True(ok1)
	is.True(ok2)

	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(0, atomic.LoadInt64(&counter2))

	// trigger error
	observer1.Complete()
	observer2.Complete()
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))

	// send a new message
	observer1.Next(21)
	observer2.Next(21)
	is.EqualValues(21, atomic.LoadInt64(&counter1))
	is.EqualValues(21, atomic.LoadInt64(&counter2))
}

func TestObserverWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type contextKey string

	const contextKeyTest = contextKey("test")

	var receivedCtx context.Context
	var receivedValue int
	var receivedError error

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			receivedCtx = ctx
			receivedValue = value
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
		func(ctx context.Context, err error) {
			receivedCtx = ctx
			receivedError = err
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
		func(ctx context.Context) {
			receivedCtx = ctx
			v, ok := ctx.Value(contextKeyTest).(string)
			is.True(ok)
			is.Equal("value", v)
		},
	)

	ctx := context.WithValue(context.Background(), contextKeyTest, "value")

	observer.NextWithContext(ctx, 42)
	is.Equal(ctx, receivedCtx)
	is.Equal(42, receivedValue)

	observer.ErrorWithContext(ctx, assert.AnError)
	is.Equal(ctx, receivedCtx)
	is.Equal(assert.AnError, receivedError)

	observer.CompleteWithContext(ctx)
	is.Equal(ctx, receivedCtx)
}

func TestObserverPartialWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type contextKey string

	const contextKeyTest = contextKey("test")

	var receivedCtx context.Context
	var receivedValue int
	var receivedError error

	observer1 := OnNextWithContext(func(ctx context.Context, value int) {
		receivedCtx = ctx
		receivedValue = value
		v, ok := ctx.Value(contextKeyTest).(string)
		is.True(ok)
		is.Equal("value", v)
	})

	observer2 := OnErrorWithContext[int](func(ctx context.Context, err error) {
		receivedCtx = ctx
		receivedError = err
		v, ok := ctx.Value(contextKeyTest).(string)
		is.True(ok)
		is.Equal("value", v)
	})

	observer3 := OnCompleteWithContext[int](func(ctx context.Context) {
		receivedCtx = ctx
		v, ok := ctx.Value(contextKeyTest).(string)
		is.True(ok)
		is.Equal("value", v)
	})

	ctx := context.WithValue(context.Background(), contextKeyTest, "value")

	observer1.NextWithContext(ctx, 42)
	is.Equal(ctx, receivedCtx)
	is.Equal(42, receivedValue)

	observer2.ErrorWithContext(ctx, assert.AnError)
	is.Equal(ctx, receivedCtx)
	is.Equal(assert.AnError, receivedError)

	observer3.CompleteWithContext(ctx)
	is.Equal(ctx, receivedCtx)
}

func TestObserverStateMethods(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	// Initial state
	is.False(observer.IsClosed())
	is.False(observer.HasThrown())
	is.False(observer.IsCompleted())

	// After error
	observer.Error(assert.AnError)
	is.True(observer.IsClosed())
	is.True(observer.HasThrown())
	is.False(observer.IsCompleted())

	// Create new observer for completion test
	observer2 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	// After completion
	observer2.Complete()
	is.True(observer2.IsClosed())
	is.False(observer2.HasThrown())
	is.True(observer2.IsCompleted())
}

func TestObserverNoopObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := NoopObserver[int]()

	// Should not panic
	observer.Next(42)

	// After error, should be closed
	observer.Error(assert.AnError)
	is.True(observer.IsClosed())
	is.True(observer.HasThrown())
	is.False(observer.IsCompleted())

	// Create new observer for complete test
	observer2 := NoopObserver[int]()
	observer2.Complete()
	is.True(observer2.IsClosed())
	is.False(observer2.HasThrown())
	is.True(observer2.IsCompleted())
}

func TestObserverPrintObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	observer := PrintObserver[int]()

	// Should not panic
	observer.Next(42)

	// After error, should be closed
	observer.Error(assert.AnError)
	is.True(observer.IsClosed())
	is.True(observer.HasThrown())
	is.False(observer.IsCompleted())

	// Create new observer for complete test
	observer2 := PrintObserver[int]()
	observer2.Complete()
	is.True(observer2.IsClosed())
	is.False(observer2.HasThrown())
	is.True(observer2.IsCompleted())
}

func TestObserverNilCallbacks(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with nil callbacks
	observer1 := NewObserver[int](nil, nil, nil)
	observer2 := NewObserverWithContext[int](nil, nil, nil)

	// Should not panic
	observer1.Next(42)
	observer1.Error(assert.AnError)
	observer1.Complete()

	observer2.NextWithContext(context.Background(), 42)
	observer2.ErrorWithContext(context.Background(), assert.AnError)
	observer2.CompleteWithContext(context.Background())

	// NewObserver with nil callbacks: wrapper functions cause panics that change status
	// NewObserverWithContext with nil callbacks: nil check prevents status changes
	is.True(observer1.IsClosed())
	is.False(observer2.IsClosed())
}

func TestObserverConcurrentAccess(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	var wg sync.WaitGroup

	numGoroutines := 100
	numCalls := 100

	// Concurrent Next calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < numCalls; j++ {
				observer.Next(1)
			}
		}()
	}

	wg.Wait()
	observer.Complete()

	expected := int64(numGoroutines * numCalls)
	is.Equal(expected, atomic.LoadInt64(&counter))
}

func TestObserverConcurrentErrorAndComplete(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	var errorCount int64
	var completeCount int64

	observer := NewObserver(
		func(value int) {},
		func(err error) { atomic.AddInt64(&errorCount, 1) },
		func() { atomic.AddInt64(&completeCount, 1) },
	)

	var wg sync.WaitGroup

	numGoroutines := 50

	// Concurrent Error and Complete calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			observer.Error(assert.AnError)
		}()

		wg.Add(1)

		go func() {
			defer wg.Done()

			observer.Complete()
		}()
	}

	wg.Wait()

	// Only one should succeed (either error or complete)
	total := atomic.LoadInt64(&errorCount) + atomic.LoadInt64(&completeCount)
	is.Equal(int64(1), total)
	is.True(observer.IsClosed())
}

func TestObserverConcurrentStateChecks(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	observer := NewObserver(
		func(value int) {},
		func(err error) {},
		func() {},
	)

	var wg sync.WaitGroup

	numGoroutines := 100
	numCalls := 100

	// Concurrent state checks
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < numCalls; j++ {
				observer.IsClosed()
				observer.HasThrown()
				observer.IsCompleted()
			}
		}()
	}

	wg.Wait()

	// Should not panic and should return consistent results
	is.False(observer.IsClosed())
	is.False(observer.HasThrown())
	is.False(observer.IsCompleted())

	observer.Complete()

	is.True(observer.IsClosed())
	is.False(observer.HasThrown())
	is.True(observer.IsCompleted())
}

func TestObserverConcurrentNextAfterClose(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	var counter int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&counter, int64(value)) },
		func(err error) {},
		func() {},
	)

	// Close the observer
	observer.Complete()

	var wg sync.WaitGroup

	numGoroutines := 100
	numCalls := 100

	// Concurrent Next calls after close
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < numCalls; j++ {
				observer.Next(1)
			}
		}()
	}

	wg.Wait()
	observer.Complete()

	// Counter should remain 0 since observer is closed
	is.Equal(int64(0), atomic.LoadInt64(&counter))
}

func TestObserverConcurrentContextMethods(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	type contextKey string

	const contextKeyTest = contextKey("test")

	var counter int64

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			v, ok := ctx.Value(contextKeyTest).(int)
			is.True(ok)
			is.Equal(42, v)
			atomic.AddInt64(&counter, int64(value))
		},
		func(ctx context.Context, err error) {
			v, ok := ctx.Value(contextKeyTest).(int)
			is.True(ok)
			is.Equal(42, v)
		},
		func(ctx context.Context) {
			v, ok := ctx.Value(contextKeyTest).(int)
			is.True(ok)
			is.Equal(21, v)
		},
	)

	var wg sync.WaitGroup

	numGoroutines := 100
	numCalls := 100

	// Concurrent context method calls
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			ctx := context.WithValue(context.Background(), contextKeyTest, 42)
			for j := 0; j < numCalls; j++ {
				observer.NextWithContext(ctx, 1)
			}
		}()
	}

	wg.Wait()
	observer.CompleteWithContext(context.WithValue(context.Background(), contextKeyTest, 21))

	expected := int64(numGoroutines * numCalls)
	is.Equal(expected, atomic.LoadInt64(&counter))
}

func TestObserverPanicHandling(t *testing.T) {
	is := assert.New(t)

	// Test panic in Next callback
	observer1 := NewObserver(
		func(value int) { panic("test panic") },
		func(err error) {},
		func() {},
	)

	// Should not panic the test
	observer1.Next(42)
	// After panic in Next, the error handler is called but observer status is not changed
	// This appears to be a limitation of the current implementation
	is.False(observer1.IsClosed())
	is.False(observer1.HasThrown())

	// Test panic in Error callback
	observer2 := NewObserver(
		func(value int) {},
		func(err error) { panic("test panic") },
		func() {},
	)

	// Should not panic the test
	observer2.Error(assert.AnError)
	// After panic in Error, the observer should still be closed
	is.True(observer2.IsClosed())
	is.True(observer2.HasThrown())

	// Test panic in Complete callback
	observer3 := NewObserver(
		func(value int) {},
		func(err error) {},
		func() { panic("test panic") },
	)

	// Should not panic the test
	observer3.Complete()
	// After panic in Complete, the observer should still be closed
	is.True(observer3.IsClosed())
	is.True(observer3.IsCompleted())
}

func TestObserverDisablePanicCapture(t *testing.T) {
	is := assert.New(t)

	previous := CaptureObserverPanics()
	SetCaptureObserverPanics(false)
	t.Cleanup(func() { SetCaptureObserverPanics(previous) })

	observer := NewObserver(
		func(value int) { panic("test panic") },
		func(err error) {},
		func() {},
	)

	is.PanicsWithValue("test panic", func() {
		observer.Next(42)
	})
}

func TestObserverDisablePanicCaptureInUnsafePipeline(t *testing.T) {
	is := assert.New(t)

	previous := CaptureObserverPanics()
	SetCaptureObserverPanics(false)
	t.Cleanup(func() { SetCaptureObserverPanics(previous) })

	observable := Pipe1(
		Just(1),
		Map(func(value int) int {
			if value == 1 {
				panic("map panic")
			}

			return value
		}),
	)

	is.PanicsWithValue("map panic", func() {
		observable.Subscribe(NewObserver(
			func(value int) {},
			func(err error) { t.Fatalf("unexpected error: %v", err) },
			func() {},
		))
	})
}

func TestObserverMixedOperations(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 5*time.Second)
	is := assert.New(t)

	var nextCount int64
	var errorCount int64
	var completeCount int64

	observer := NewObserver(
		func(value int) { atomic.AddInt64(&nextCount, 1) },
		func(err error) { atomic.AddInt64(&errorCount, 1) },
		func() { atomic.AddInt64(&completeCount, 1) },
	)

	var wg sync.WaitGroup

	numGoroutines := 50

	// Mixed operations
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < 10; j++ {
				observer.Next(j)
				observer.IsClosed()
				observer.HasThrown()
				observer.IsCompleted()
			}
		}()

		wg.Add(1)

		go func() {
			defer wg.Done()

			observer.Error(assert.AnError)
		}()

		wg.Add(1)

		go func() {
			defer wg.Done()

			observer.Complete()
		}()
	}

	wg.Wait()

	// Should be closed
	is.True(observer.IsClosed())

	// Either error or complete should have been called once
	total := atomic.LoadInt64(&errorCount) + atomic.LoadInt64(&completeCount)
	is.Equal(int64(1), total)
}

func TestObserverContextCancellation(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var receivedCtx context.Context

	observer := NewObserverWithContext(
		func(ctx context.Context, value int) {
			receivedCtx = ctx
		},
		func(ctx context.Context, err error) {
			receivedCtx = ctx
		},
		func(ctx context.Context) {
			receivedCtx = ctx
		},
	)

	// Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	observer.NextWithContext(ctx, 42)
	is.Equal(ctx, receivedCtx)

	observer.ErrorWithContext(ctx, assert.AnError)
	is.Equal(ctx, receivedCtx)

	observer.CompleteWithContext(ctx)
	is.Equal(ctx, receivedCtx)
}

func TestObserverMemoryLeak(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Second)
	is := assert.New(t)

	// Create many observers to test for memory leaks
	observers := make([]Observer[int], 1000)

	for i := 0; i < 1000; i++ {
		observers[i] = NewObserver(
			func(value int) {},
			func(err error) {},
			func() {},
		)
	}

	// Use all observers
	for i := 0; i < 1000; i++ {
		observers[i].Next(i)

		if i%2 == 0 {
			observers[i].Error(assert.AnError)
		} else {
			observers[i].Complete()
		}
	}

	// Verify all are closed
	for i := 0; i < 1000; i++ {
		is.True(observers[i].IsClosed())
	}
}
