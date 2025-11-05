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
		status: 0,

		observers:     sync.Map{},
		observerIndex: 0,

		hasValue: false,
		value:    lo.T2(context.TODO(), lo.Empty[T]()),
		err:      lo.T2[context.Context, error](context.TODO(), nil),
	}
}

type asyncSubjectImpl[T any] struct {
	status Kind

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

	switch s.status {
	case KindNext:
		// fallthrough
	case KindError:
		subscription.ErrorWithContext(s.err.A, s.err.B)
		return subscription
	case KindComplete:
		if s.hasValue {
			subscription.NextWithContext(s.value.A, s.value.B)
		}

		subscription.CompleteWithContext(subscriberCtx)

		return subscription
	}

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, subscription)

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
	if s.status == KindNext {
		s.hasValue = true
		s.value = lo.T2(ctx, value) // A previous value might be erased. It won't be forwarded to `OnDroppedNotification`.
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(value))
	}
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	if s.status == KindNext {
		s.err = lo.T2(ctx, err)
		s.status = KindError
		s.broadcastError(ctx, err)
	} else {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
	}

	s.unsubscribeAll()
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.status == KindNext {
		s.status = KindComplete
		if s.hasValue {
			s.broadcastNext(s.value.A, s.value.B)
		}

		s.broadcastComplete(ctx)
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.unsubscribeAll()
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
	return s.status != KindNext
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) HasThrown() bool {
	return s.status == KindError
}

// Implements Observer.
func (s *asyncSubjectImpl[T]) IsCompleted() bool {
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
