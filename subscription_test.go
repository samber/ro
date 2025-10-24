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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestSubscriptionNewSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with nil teardown
	sub := NewSubscription(nil)
	is.NotNil(sub)
	is.False(sub.IsClosed())

	// Test with teardown function
	called := false
	teardown := func() {
		called = true
	}
	sub = NewSubscription(teardown)
	is.NotNil(sub)
	is.False(sub.IsClosed())
	is.False(called)

	// Test immediate execution when subscription is already closed
	sub.Unsubscribe()
	is.True(sub.IsClosed())
	is.True(called)
}

func TestSubscriptionAdd(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)
	called := false
	teardown := func() {
		called = true
	}

	// Test adding teardown to active subscription
	sub.Add(teardown)
	is.False(called)
	is.False(sub.IsClosed())

	// Test adding nil teardown
	sub.Add(nil)
	is.False(called)

	// Test immediate execution when subscription is closed
	sub.Unsubscribe()
	is.True(called)
	is.True(sub.IsClosed())

	// Test adding teardown to already closed subscription
	called2 := false
	teardown2 := func() {
		called2 = true
	}
	sub.Add(teardown2)
	is.True(called2) // Should be called immediately
}

func TestSubscriptionAddUnsubscribable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)
	called := false
	unsubscribable := &mockUnsubscribable{
		unsubscribe: func() {
			called = true
		},
	}

	// Test adding unsubscribable to active subscription
	sub.AddUnsubscribable(unsubscribable)
	is.False(called)
	is.False(sub.IsClosed())

	// Test adding nil unsubscribable
	sub.AddUnsubscribable(nil)
	is.False(called)

	// Test execution when subscription is closed
	sub.Unsubscribe()
	is.True(called)
	is.True(sub.IsClosed())

	// Test adding to already closed subscription
	called2 := false
	unsubscribable2 := &mockUnsubscribable{
		unsubscribe: func() {
			called2 = true
		},
	}
	sub.AddUnsubscribable(unsubscribable2)
	is.True(called2) // Should be called immediately
}

func TestSubscriptionUnsubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test single teardown
	called := false
	teardown := func() {
		called = true
	}
	sub := NewSubscription(teardown)

	sub.Unsubscribe()
	is.True(called)
	is.True(sub.IsClosed())

	// Test multiple teardowns
	called1, called2, called3 := false, false, false
	teardown1 := func() { called1 = true }
	teardown2 := func() { called2 = true }
	teardown3 := func() { called3 = true }

	sub2 := NewSubscription(teardown1)
	sub2.Add(teardown2)
	sub2.Add(teardown3)

	sub2.Unsubscribe()
	is.True(called1)
	is.True(called2)
	is.True(called3)
	is.True(sub2.IsClosed())

	// Test double unsubscribe
	sub2.Unsubscribe() // Should not panic or cause issues
	is.True(sub2.IsClosed())
}

func TestSubscriptionIsClosed(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)
	is.False(sub.IsClosed())

	sub.Unsubscribe()
	is.True(sub.IsClosed())

	// Test after double unsubscribe
	sub.Unsubscribe()
	is.True(sub.IsClosed())
}

func TestSubscriptionWait(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	// Test that Wait blocks until unsubscribe
	done := make(chan bool, 1)

	go func() {
		sub.Wait()

		done <- true
	}()

	// Give some time for the goroutine to start
	time.Sleep(10 * time.Millisecond)

	// The channel should not have received anything yet
	select {
	case <-done:
		is.Fail("Wait should block until unsubscribe")
	default:
		// Expected - Wait is blocking
	}

	// Unsubscribe should unblock Wait
	sub.Unsubscribe()

	// Wait for the goroutine to complete
	select {
	case <-done:
		// Expected
	case <-time.After(100 * time.Millisecond):
		is.Fail("Wait should unblock after unsubscribe")
	}
}

func TestSubscriptionPanicHandling(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test teardown that panics
	panicTeardown := func() {
		panic("test panic")
	}
	sub := NewSubscription(panicTeardown)

	// Should panic when unsubscribe is called
	is.Panics(func() {
		sub.Unsubscribe()
	})

	// Test multiple teardowns with one that panics
	called := false
	normalTeardown := func() {
		called = true
	}

	sub2 := NewSubscription(normalTeardown)
	sub2.Add(panicTeardown)

	// Should panic, but normal teardown should still be called
	is.Panics(func() {
		sub2.Unsubscribe()
	})
	is.True(called)
}

func TestSubscriptionConcurrentAdd(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	var wg sync.WaitGroup

	counter := int32(0)

	// Add teardowns concurrently
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Add(func() {
				atomic.AddInt32(&counter, 1)
			})
		}()
	}

	wg.Wait()
	is.False(sub.IsClosed())

	// Unsubscribe should execute all teardowns
	sub.Unsubscribe()
	is.Equal(int32(100), counter)
	is.True(sub.IsClosed())
}

func TestSubscriptionConcurrentUnsubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)
	counter := int32(0)

	// Add some teardowns
	for i := 0; i < 10; i++ {
		sub.Add(func() {
			atomic.AddInt32(&counter, 1)
		})
	}

	var wg sync.WaitGroup
	// Call unsubscribe concurrently
	for i := 0; i < 10; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Unsubscribe()
		}()
	}

	wg.Wait()

	// Should only execute teardowns once
	is.Equal(int32(10), counter)
	is.True(sub.IsClosed())
}

func TestSubscriptionConcurrentAddAndUnsubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	var wg sync.WaitGroup

	counter := int32(0)

	// Start goroutines that add teardowns
	for i := 0; i < 50; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Add(func() {
				atomic.AddInt32(&counter, 1)
			})
		}()
	}

	// Start goroutines that unsubscribe
	for i := 0; i < 10; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Unsubscribe()
		}()
	}

	wg.Wait()

	// Should be closed
	is.True(sub.IsClosed())
	// Counter should be at least 1 (from the initial teardowns)
	is.GreaterOrEqual(counter, int32(1))
}

func TestSubscriptionConcurrentIsClosed(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	var wg sync.WaitGroup

	closedCount := int32(0)

	// Start goroutines that check IsClosed
	for i := 0; i < 100; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			if sub.IsClosed() {
				atomic.AddInt32(&closedCount, 1)
			}
		}()
	}

	// Start a goroutine that unsubscribes
	wg.Add(1)

	go func() {
		defer wg.Done()

		time.Sleep(10 * time.Millisecond)
		sub.Unsubscribe()
	}()

	wg.Wait()

	// Should be closed
	is.True(sub.IsClosed())
}

func TestSubscriptionConcurrentWait(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	var wg sync.WaitGroup

	waitCount := int32(0)

	// Start multiple goroutines that wait
	for i := 0; i < 10; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Wait()
			atomic.AddInt32(&waitCount, 1)
		}()
	}

	// Give time for goroutines to start
	time.Sleep(10 * time.Millisecond)

	// All should still be waiting
	is.Equal(int32(0), waitCount)

	// Unsubscribe should unblock all waiters
	sub.Unsubscribe()

	// Wait for all goroutines to complete
	wg.Wait()

	// All should have completed
	is.Equal(int32(10), waitCount)
}

func TestSubscriptionMixedOperations(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)

	var wg sync.WaitGroup

	counter := int32(0)

	// Mix of operations
	for i := 0; i < 20; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.Add(func() {
				atomic.AddInt32(&counter, 1)
			})
		}()

		wg.Add(1)

		go func() {
			defer wg.Done()

			sub.IsClosed()
		}()

		if i%5 == 0 {
			wg.Add(1)

			go func() {
				defer wg.Done()

				sub.Unsubscribe()
			}()
		}
	}

	wg.Wait()

	// Should be closed
	is.True(sub.IsClosed())
}

func TestSubscriptionErrorHandling(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test error in teardown
	errorTeardown := func() {
		panic(errors.New("test error"))
	}
	sub := NewSubscription(errorTeardown)

	// Should panic with the error
	is.Panics(func() {
		sub.Unsubscribe()
	})

	// Test multiple teardowns with errors
	normalCalled := false
	normalTeardown := func() {
		normalCalled = true
	}

	sub2 := NewSubscription(normalTeardown)
	sub2.Add(errorTeardown)

	// Should panic, but normal teardown should still be called
	is.Panics(func() {
		sub2.Unsubscribe()
	})
	is.True(normalCalled)
}

func TestSubscriptionNilHandling(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with nil teardown
	sub := NewSubscription(nil)
	is.NotNil(sub)

	// Should not panic
	sub.Add(nil)
	sub.AddUnsubscribable(nil)
	sub.Unsubscribe()

	// Should still be closed
	is.True(sub.IsClosed())
}

func TestSubscriptionMemoryLeak(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test that finalizers are cleared after unsubscribe
	sub := NewSubscription(nil)

	// Add many teardowns
	for i := 0; i < 1000; i++ {
		sub.Add(func() {})
	}

	// Unsubscribe should clear the finalizers
	sub.Unsubscribe()

	// Adding more should execute immediately
	called := false

	sub.Add(func() {
		called = true
	})
	is.True(called)
}

// Mock implementation for testing.
type mockUnsubscribable struct {
	unsubscribe func()
}

func (m *mockUnsubscribable) Unsubscribe() {
	if m.unsubscribe != nil {
		m.unsubscribe()
	}
}
func (m *mockUnsubscribable) UnsubscribeWithContext(ctx context.Context) {
	if m.unsubscribe != nil {
		m.unsubscribe()
	}
}


