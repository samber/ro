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

	"github.com/samber/lo"
)

var _ Subject[int] = (*asyncSubjectImpl[int])(nil)

// NewAsyncSubject emits its last value on completion. If no value had been received, observer is completed without emitting value.
// The emitted value or error is stored for future subscriptions. 0 or 1 value is emitted.
func NewAsyncSubject[T any]() Subject[T] {
	return &asyncSubjectImpl[T]{
		mu:     sync.Mutex{},
		queue:  notificationQueue{},
		status: 0,

		observers:     sync.Map{},
		observerIndex: 0,

		hasValue: false,
		value:    lo.T2(context.TODO(), lo.Empty[T]()),
		err:      lo.T2[context.Context, error](context.TODO(), nil),
	}
}

type asyncSubjectImpl[T any] struct {
	mu     sync.Mutex // sync.RWMutex would be better, but it is too slow for high-volume subjects
	status Kind

	// queue delivers notifications outside mu, in order, and queues re-entrant calls.
	queue notificationQueue

	observers     sync.Map
	observerIndex uint32

	hasValue bool
	value    lo.Tuple2[context.Context, T]
	err      lo.Tuple2[context.Context, error]
}

// Implements Observable.
func (s *asyncSubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *asyncSubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriber(destination)

	s.mu.Lock()

	// The lock is never held while user code runs (observer callbacks, teardowns):
	// a callback calling back into the subject would deadlock.
	switch s.status {
	case KindNext:
		// fallthrough
	case KindError:
		errCtx, err := s.err.A, s.err.B
		s.mu.Unlock()

		subscription.ErrorWithContext(errCtx, err)

		return subscription
	case KindComplete:
		hasValue, value := s.hasValue, s.value
		s.mu.Unlock()

		if hasValue {
			subscription.NextWithContext(value.A, value.B)
		}

		subscription.CompleteWithContext(subscriberCtx)

		return subscription
	}

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, subscription)
	s.mu.Unlock()

	subscription.Add(func() {
		s.observers.Delete(index)
	})

	return subscription
}

func (s *asyncSubjectImpl[T]) unsubscribeAll() {
	s.observers.Range(func(key, observer any) bool {
		s.observers.Delete(key)
		return true
	})
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	s.queue.run(&s.mu, func() func() {
		if s.status != KindNext {
			return func() { OnDroppedNotification(ctx, NewNotificationNext(value)) }
		}

		s.hasValue = true
		s.value = lo.T2(ctx, value) // A previous value might be erased. It won't be forwarded to `OnDroppedNotification`.

		return nil
	})
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	s.queue.run(&s.mu, func() func() {
		if s.status != KindNext {
			return func() { OnDroppedNotification(ctx, NewNotificationError[T](err)) }
		}

		s.err = lo.T2(ctx, err)
		s.status = KindError

		return func() {
			s.broadcastError(ctx, err)
			s.unsubscribeAll()
		}
	})
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	s.queue.run(&s.mu, func() func() {
		if s.status != KindNext {
			return func() { OnDroppedNotification(ctx, NewNotificationComplete[T]()) }
		}

		hasValue, value := s.hasValue, s.value
		s.status = KindComplete

		return func() {
			if hasValue {
				s.broadcastNext(value.A, value.B)
			}

			s.broadcastComplete(ctx)
			s.unsubscribeAll()
		}
	})
}

func (s *asyncSubjectImpl[T]) HasObserver() bool {
	has := false

	s.observers.Range(func(key, value any) bool {
		has = true
		return false
	})

	return has
}

func (s *asyncSubjectImpl[T]) CountObservers() int {
	count := 0

	s.observers.Range(func(key, value any) bool {
		count++
		return true
	})

	return count
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) IsClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status != KindNext
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) HasThrown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindError
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) IsCompleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindComplete
}

func (s *asyncSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *asyncSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *asyncSubjectImpl[T]) broadcastNext(ctx context.Context, value T) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).NextWithContext(ctx, value) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *asyncSubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *asyncSubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
