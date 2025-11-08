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
	"testing"

	"github.com/samber/ro/internal/xsync"
	"github.com/stretchr/testify/assert"
)

func TestSubscriberImpl_ErrorWithContext_locklessNilDestination(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create a lockless subscriber with nil destination
	subscriber := &subscriberImpl[int]{
		status:       0,
		backpressure: BackpressureBlock,
		mu:           nil,
		destination:  nil,
		Subscription: NewSubscription(nil),
		mode:         ConcurrencyModeSingleProducer,
		lockless:     true,
	}

	// Should handle nil destination gracefully
	subscriber.ErrorWithContext(context.Background(), assert.AnError)
	is.Equal(int32(1), subscriber.status) // Should transition to error state
}

func TestSubscriberImpl_CompleteWithContext_locklessNilDestination(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create a lockless subscriber with nil destination
	subscriber := &subscriberImpl[int]{
		status:       0,
		backpressure: BackpressureBlock,
		mu:           nil,
		destination:  nil,
		Subscription: NewSubscription(nil),
		mode:         ConcurrencyModeSingleProducer,
		lockless:     true,
	}

	// Should handle nil destination gracefully
	subscriber.CompleteWithContext(context.Background())
	is.Equal(int32(2), subscriber.status) // Should transition to complete state
}

func TestSubscriberImpl_setDirectors_nonObserverImpl(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create a custom observer that doesn't use observerImpl
	type customObserver struct {
		nextCalled     bool
		errorCalled    bool
		completeCalled bool
	}

	custom := &customObserver{}

	// Create an observer wrapper
	observer := NewObserver(
		func(value int) { custom.nextCalled = true },
		func(err error) { custom.errorCalled = true },
		func() { custom.completeCalled = true },
	)

	subscriber := &subscriberImpl[int]{
		status:       0,
		backpressure: BackpressureBlock,
		mu:           xsync.NewMutexWithLock(),
		destination:  observer,
		Subscription: NewSubscription(nil),
		mode:         ConcurrencyModeSafe,
		lockless:     false,
	}

	// Call setDirectors with a non-observerImpl destination
	// This should set up the default interface-based calls
	subscriber.setDirectors(observer, true)

	// Verify directors were set
	is.NotNil(subscriber.nextDirect)
	is.NotNil(subscriber.errorDirect)
	is.NotNil(subscriber.completeDirect)

	// Test that the directors work
	subscriber.nextDirect(context.Background(), 1)
	is.True(custom.nextCalled)

	subscriber.errorDirect(context.Background(), assert.AnError)
	is.True(custom.errorCalled)

	subscriber.completeDirect(context.Background())
	is.True(custom.completeCalled)
}

func TestSubscriberImpl_setDirectors_withObserverImpl(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var nextCalled, errorCalled, completeCalled bool

	// Create an observerImpl directly
	observer := &observerImpl[int]{
		status:        0,
		capturePanics: true,
		onNext:        func(ctx context.Context, value int) { nextCalled = true },
		onError:       func(ctx context.Context, err error) { errorCalled = true },
		onComplete:    func(ctx context.Context) { completeCalled = true },
	}

	subscriber := &subscriberImpl[int]{
		status:       0,
		backpressure: BackpressureBlock,
		mu:           xsync.NewMutexWithLock(),
		destination:  observer,
		Subscription: NewSubscription(nil),
		mode:         ConcurrencyModeSafe,
		lockless:     false,
	}

	// Call setDirectors with an observerImpl destination
	// This should set up the optimized tryXXXWithCapture calls
	subscriber.setDirectors(observer, true)

	// Verify directors were set
	is.NotNil(subscriber.nextDirect)
	is.NotNil(subscriber.errorDirect)
	is.NotNil(subscriber.completeDirect)

	// Test that the directors work and use the optimized path
	subscriber.nextDirect(context.Background(), 1)
	is.True(nextCalled)

	subscriber.errorDirect(context.Background(), assert.AnError)
	is.True(errorCalled)

	subscriber.completeDirect(context.Background())
	is.True(completeCalled)
}

func TestSubscriberImpl_setDirectors_noCapture_propagatesPanics(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create an observerImpl whose handlers panic. When setDirectors is
	// invoked with capture==false we expect these panics to propagate
	// (i.e. no TryCatch wrapper should be used).
	observer := &observerImpl[int]{
		status:        0,
		capturePanics: true,
		onNext:        func(ctx context.Context, value int) { panic("onNext panic") },
		onError:       func(ctx context.Context, err error) { panic("onError panic") },
		onComplete:    func(ctx context.Context) { panic("onComplete panic") },
	}

	subscriber := &subscriberImpl[int]{
		status:       0,
		backpressure: BackpressureBlock,
		mu:           xsync.NewMutexWithLock(),
		destination:  observer,
		Subscription: NewSubscription(nil),
		mode:         ConcurrencyModeSafe,
		lockless:     false,
	}

	// Configure directors with capture=false so the direct helpers call
	// the internal onNext/onError/onComplete without TryCatch wrappers.
	subscriber.setDirectors(observer, false)

	is.NotNil(subscriber.nextDirect)
	is.NotNil(subscriber.errorDirect)
	is.NotNil(subscriber.completeDirect)

	// Each direct call should panic (propagate) because capture==false.
	is.Panics(func() { subscriber.nextDirect(context.Background(), 1) })
	is.Panics(func() { subscriber.errorDirect(context.Background(), assert.AnError) })
	is.Panics(func() { subscriber.completeDirect(context.Background()) })
}
