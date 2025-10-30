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

var _ Subject[int] = (*publishSubjectImpl[int])(nil)

// NewPublishSubject broadcasts a value to observers (fanout).
// Values received before subscription are not transmitted.
func NewPublishSubject[T any]() Subject[T] {
	return &publishSubjectImpl[T]{
		status: KindNext,

		observers:     sync.Map{},
		observerIndex: 0,

		err: lo.Tuple2[context.Context, error]{},
	}
}

type publishSubjectImpl[T any] struct {
	status Kind

	observers     sync.Map
	observerIndex uint32

	err lo.Tuple2[context.Context, error]
}

// Implements Observable.
func (s *publishSubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *publishSubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
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

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, subscription)

	subscription.Add(func() {
		s.observers.Delete(index)
	})

	return subscription
}

func (s *publishSubjectImpl[T]) unsubscribeAll() {
	s.observers.Range(func(key, _ any) bool {
		s.observers.Delete(key)
		return true
	})
}

// Implements Observer.
func (s *publishSubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *publishSubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	if s.status == KindNext {
		s.broadcastNext(ctx, value)
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(value))
	}
}

// Implements Observer.
func (s *publishSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *publishSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
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
func (s *publishSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *publishSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.status == KindNext {
		s.status = KindComplete
		s.broadcastComplete(ctx)
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.unsubscribeAll()
}

func (s *publishSubjectImpl[T]) HasObserver() (has bool) {
	has = false

	s.observers.Range(func(key, value any) bool {
		has = true
		return false
	})

	return has
}

func (s *publishSubjectImpl[T]) CountObservers() int {
	count := 0

	s.observers.Range(func(key, value any) bool {
		count++
		return true
	})

	return count
}

// Implements Observer.
func (s *publishSubjectImpl[T]) IsClosed() bool {
	return s.status != KindNext
}

// Implements Observer.
func (s *publishSubjectImpl[T]) HasThrown() bool {
	return s.status == KindError
}

// Implements Observer.
func (s *publishSubjectImpl[T]) IsCompleted() bool {
	return s.status == KindComplete
}

func (s *publishSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *publishSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *publishSubjectImpl[T]) broadcastNext(ctx context.Context, value T) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).NextWithContext(ctx, value) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *publishSubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *publishSubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
