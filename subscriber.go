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
	"sync/atomic"

	"github.com/samber/ro/internal/xsync"
)

// Subscriber implements the Observer and Subscription interfaces. While the Observer is
// the public API for consuming the values of an Observable, all Observers get
// converted to a Subscriber, in order to provide Subscription-like capabilities
// such as `Unsubscribe()`. Subscriber is a common type in samber/ro, and crucial for
// implementing operators, but it is rarely used as a public API.
type Subscriber[T any] interface {
	Subscription
	Observer[T]
}

var _ Subscriber[int] = (*subscriberImpl[int])(nil)

// NewSubscriber creates a new Subscriber from an Observer. If the Observer
// is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// This method is safe for concurrent use.
//
// It is rarely used as a public API.
func NewSubscriber[T any](destination Observer[T]) Subscriber[T] {
	return NewSafeSubscriber(destination)
}

// NewSafeSubscriber creates a new Subscriber from an Observer. If the Observer
// is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// This method is safe for concurrent use.
//
// It is rarely used as a public API.
func NewSafeSubscriber[T any](destination Observer[T]) Subscriber[T] {
	return NewSubscriberWithConcurrencyMode(destination, ConcurrencyModeSafe)
}

// NewUnsafeSubscriber creates a new Subscriber from an Observer. If the Observer
// is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// This method is not safe for concurrent use.
//
// It is rarely used as a public API.
func NewUnsafeSubscriber[T any](destination Observer[T]) Subscriber[T] {
	return NewSubscriberWithConcurrencyMode(destination, ConcurrencyModeUnsafe)
}

// NewEventuallySafeSubscriber creates a new Subscriber from an Observer. If the Observer
// is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// This method is safe for concurrent use, but concurrent messages are dropped.
//
// It is rarely used as a public API.
func NewEventuallySafeSubscriber[T any](destination Observer[T]) Subscriber[T] {
	return NewSubscriberWithConcurrencyMode(destination, ConcurrencyModeEventuallySafe)
}

// NewSingleProducerSubscriber creates a new Subscriber optimized for single producer scenarios.
// If the Observer is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// This method is not safe for concurrent producers.
func NewSingleProducerSubscriber[T any](destination Observer[T]) Subscriber[T] {
	return NewSubscriberWithConcurrencyMode(destination, ConcurrencyModeSingleProducer)
}

// NewSubscriberWithConcurrencyMode creates a new Subscriber from an Observer. If the Observer
// is already a Subscriber, it is returned as is. Otherwise, a new Subscriber
// is created that wraps the Observer.
//
// The returned Subscriber will unsubscribe from the destination Observer when
// Unsubscribe() is called.
//
// It is rarely used as a public API.
func NewSubscriberWithConcurrencyMode[T any](destination Observer[T], mode ConcurrencyMode) Subscriber[T] {
	// Spinlock is ignored because it is too slow when chaining operators. Spinlock should be used
	// only for short-lived local locks.
	switch mode {
	case ConcurrencyModeSafe:
		// Fully synchronized subscriber that uses a real mutex implementation.
		return newSubscriberImpl(mode, xsync.NewMutexWithLock(), BackpressureBlock, destination, false)
	case ConcurrencyModeUnsafe:
		// Unsafe mode: uses a no-op mutex object. Method calls to Lock/Unlock will be executed
		// but they are no-ops; this preserves the same call-site shape as the safe variant while
		// avoiding actual synchronization overhead.
		return newSubscriberImpl(mode, xsync.NewMutexWithoutLock(), BackpressureBlock, destination, false)
	case ConcurrencyModeEventuallySafe:
		// Safe with backpressure drop: uses a real mutex but drops values when the lock cannot
		// be acquired immediately.
		return newSubscriberImpl(mode, xsync.NewMutexWithLock(), BackpressureDrop, destination, false)
	case ConcurrencyModeSingleProducer:
		// Single-producer optimized: uses the lockless fast path (mu == nil, lockless == true).
		// This avoids any Lock/Unlock calls on the hot path and relies on atomics for status
		// checks. It is intentionally different from ConcurrencyModeUnsafe which still calls
		// no-op Lock/Unlock methods (and therefore incurs a method call per notification).
		return newSubscriberImpl(mode, nil, BackpressureBlock, destination, true)
	default:
		panic("invalid concurrency mode")
	}
}

// newSubscriberImpl creates a new subscriber implementation with the specified
// synchronization behavior and destination observer.
func newSubscriberImpl[T any](mode ConcurrencyMode, mu xsync.Mutex, backpressure Backpressure, destination Observer[T], lockless bool) Subscriber[T] {
	// Protect against multiple encapsulation layers.
	if subscriber, ok := destination.(Subscriber[T]); ok {
		return subscriber
	}

	// Note: `mu == nil` combined with `lockless == true` enables the fast-path used by
	// `ConcurrencyModeSingleProducer` where the subscriber avoids calling Lock/Unlock on each
	// notification and instead uses atomic status checks. `xsync.NewMutexWithoutLock()` is a
	// no-op mutex implementation used by `ConcurrencyModeUnsafe` — its Lock/Unlock methods are
	// still invoked but do nothing. We keep both variants to make the performance trade-offs
	// explicit and measurable.

	subscriber := &subscriberImpl[T]{
		status:       0, // KindNext
		backpressure: backpressure,

		mu:          mu,
		destination: destination,

		Subscription: NewSubscription(nil),
		mode:         mode,
		lockless:     lockless,
	}

	if subscription, ok := destination.(Subscription); ok {
		subscription.Add(subscriber.Unsubscribe)
	}

	return subscriber
}

type subscriberImpl[T any] struct {
	// While mutex is used for synchronization of producer, status is used for storing state of
	// the subscriber. Using the mutex for reading the status would have create a dead lock if
	// an Observer calls Unsubscribe(), IsClosed(), HasThrown(), IsCompleted() synchronously.
	//
	// 0 - KindNext
	// 1 - KindError
	// 2 - KindComplete
	status       int32
	backpressure Backpressure

	_ [59]byte // padding to prevent false sharing

	// Mutex are much much faster than channels.
	//
	// Also, generators has been added in go1.23. A different implem of Observable/Observer
	// might reduce latency induced by mutexes.
	//
	// It could be interesting to implement a lock-free version of this,
	// with message drop instead of backpressure, and when SLO must be kept under
	// control (real-time streams?).
	mu          xsync.Mutex
	destination Observer[T]

	Subscription

	mode     ConcurrencyMode
	lockless bool
}

// Implements Observer.
func (s *subscriberImpl[T]) Next(v T) {
	s.NextWithContext(context.Background(), v)
}

// Implements Observer.
func (s *subscriberImpl[T]) NextWithContext(ctx context.Context, v T) {
	if s.destination == nil {
		return
	}

	if s.lockless {
		if atomic.LoadInt32(&s.status) == 0 {
			s.destination.NextWithContext(ctx, v)
		} else {
			OnDroppedNotification(ctx, NewNotificationNext(v))
		}

		return
	}

	if s.backpressure == BackpressureDrop {
		if !s.mu.TryLock() {
			OnDroppedNotification(ctx, NewNotificationNext(v))
			return
		}
	} else {
		s.mu.Lock()
	}

	if atomic.LoadInt32(&s.status) == 0 {
		s.destination.NextWithContext(ctx, v)
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(v))
	}

	s.mu.Unlock()
}

// Implements Observer.
func (s *subscriberImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *subscriberImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	if s.lockless {
		if atomic.CompareAndSwapInt32(&s.status, 0, 1) {
			if s.destination != nil {
				s.destination.ErrorWithContext(ctx, err)
			}
		} else {
			OnDroppedNotification(ctx, NewNotificationError[T](err))
		}

		s.unsubscribe()

		return
	}

	s.mu.Lock()

	if atomic.CompareAndSwapInt32(&s.status, 0, 1) {
		if s.destination != nil {
			s.destination.ErrorWithContext(ctx, err)
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
	}

	s.mu.Unlock()

	s.unsubscribe()
}

// Implements Observer.
func (s *subscriberImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *subscriberImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.lockless {
		if atomic.CompareAndSwapInt32(&s.status, 0, 2) {
			if s.destination != nil {
				s.destination.CompleteWithContext(ctx)
			}
		} else {
			OnDroppedNotification(ctx, NewNotificationComplete[T]())
		}

		s.unsubscribe()

		return
	}

	s.mu.Lock()

	if atomic.CompareAndSwapInt32(&s.status, 0, 2) {
		if s.destination != nil {
			s.destination.CompleteWithContext(ctx)
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.mu.Unlock()

	s.unsubscribe()
}

// Implements Observer.
func (s *subscriberImpl[T]) IsClosed() bool {
	return atomic.LoadInt32(&s.status) != 0
}

// Implements Observer.
func (s *subscriberImpl[T]) HasThrown() bool {
	return atomic.LoadInt32(&s.status) == 1
}

// Implements Observer.
func (s *subscriberImpl[T]) IsCompleted() bool {
	return atomic.LoadInt32(&s.status) == 2
}

// Implements Observer.
func (s *subscriberImpl[T]) Unsubscribe() {
	if atomic.CompareAndSwapInt32(&s.status, 0, 2) {
		s.unsubscribe()
	}
}

func (s *subscriberImpl[T]) unsubscribe() {
	// s.Subscription.Unsubscribe() is protected against concurrent calls.
	s.Subscription.Unsubscribe()
}
