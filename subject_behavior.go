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

var _ Subject[int] = (*behaviorSubjectImpl[int])(nil)

// NewBehaviorSubject emits the current value to new subscribers or initial value.
// After completion, new subscription won't receive the last value, but the error will eventually propagated.
func NewBehaviorSubject[T any](initial T) Subject[T] {
	return &behaviorSubjectImpl[T]{
		status: KindNext,

		observers:     sync.Map{},
		observerIndex: 0,

		last: lo.T2(context.TODO(), initial),
		err:  lo.Tuple2[context.Context, error]{},
	}
}

type behaviorSubjectImpl[T any] struct {
	status Kind

	observers     sync.Map
	observerIndex uint32

	last lo.Tuple2[context.Context, T]
	err  lo.Tuple2[context.Context, error]
}

// Implements Observable.
func (s *behaviorSubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *behaviorSubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriber(destination)

	switch s.status {
	case KindNext:
		// fallthrough
	case KindError:
		subscription.ErrorWithContext(s.err.A, s.err.B)
		return subscription
	case KindComplete:
		subscription.CompleteWithContext(subscriberCtx)
		return subscription
	}

	// until we get a first value, should we send subscriberCtx or last.A (== context.TODO()) ?
	subscription.NextWithContext(s.last.A, s.last.B)

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, subscription)

	subscription.Add(func() {
		s.observers.Delete(index)
	})

	return subscription
}

func (s *behaviorSubjectImpl[T]) unsubscribeAll() {
	s.observers.Range(func(key, value any) bool {
		s.observers.Delete(key)
		return true
	})
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	if s.status == KindNext {
		s.last = lo.T2(ctx, value)
		s.broadcastNext(ctx, value)
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(value))
	}
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
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
func (s *behaviorSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.status == KindNext {
		s.status = KindComplete
		s.broadcastComplete(ctx)
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.unsubscribeAll()
}

func (s *behaviorSubjectImpl[T]) HasObserver() (has bool) {
	has = false

	s.observers.Range(func(key, value any) bool {
		has = true
		return false
	})

	return has
}

func (s *behaviorSubjectImpl[T]) CountObservers() int {
	count := 0

	s.observers.Range(func(key, value any) bool {
		count++
		return true
	})

	return count
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) IsClosed() bool {
	return s.status != KindNext
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) HasThrown() bool {
	return s.status == KindError
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) IsCompleted() bool {
	return s.status == KindComplete
}

func (s *behaviorSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *behaviorSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *behaviorSubjectImpl[T]) broadcastNext(ctx context.Context, value T) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).NextWithContext(ctx, value) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *behaviorSubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *behaviorSubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
