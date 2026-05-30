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

// ReplaySubjectUnlimitedBufferSize is the unlimited buffer size for a ReplaySubject.
const ReplaySubjectUnlimitedBufferSize = -1

var _ Subject[int] = (*replaySubjectImpl[int])(nil)

// NewReplaySubject emits old values to new subscribers.
// After error or completion, new subscriptions receive values from the buffer then the error or the completion.
func NewReplaySubject[T any](bufferSize int) Subject[T] {
	return &replaySubjectImpl[T]{
		status: KindNext,

		observers:     sync.Map{},
		observerIndex: 0,

		err:        lo.Tuple2[context.Context, error]{},
		values:     []lo.Tuple2[context.Context, T]{},
		bufferSize: bufferSize,
	}
}

type replaySubjectImpl[T any] struct {
	status Kind

	observers     sync.Map
	observerIndex uint32

	err        lo.Tuple2[context.Context, error]
	values     []lo.Tuple2[context.Context, T]
	bufferSize int
}

// Implements Observable.
func (s *replaySubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *replaySubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriber(destination)

	for _, v := range s.values {
		subscription.NextWithContext(v.A, v.B)
	}

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

func (s *replaySubjectImpl[T]) unsubscribeAll() {
	s.observers.Range(func(key, value any) bool {
		s.observers.Delete(key)
		return true
	})
}

// Implements Observer.
func (s *replaySubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *replaySubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	if s.status == KindNext {
		s.broadcastNext(ctx, value)

		s.values = append(s.values, lo.T2(ctx, value))
		if s.bufferSize != ReplaySubjectUnlimitedBufferSize && len(s.values) > s.bufferSize {
			OnDroppedNotification(ctx, NewNotificationNext(s.values[0].B))
			s.values = s.values[len(s.values)-s.bufferSize:]
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(value))
	}
}

// Implements Observer.
func (s *replaySubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *replaySubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
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
func (s *replaySubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *replaySubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.status == KindNext {
		s.status = KindComplete
		s.broadcastComplete(ctx)
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.unsubscribeAll()
}

func (s *replaySubjectImpl[T]) HasObserver() bool {
	has := false

	s.observers.Range(func(key, value any) bool {
		has = true
		return false
	})

	return has
}

func (s *replaySubjectImpl[T]) CountObservers() int {
	count := 0

	s.observers.Range(func(key, value any) bool {
		count++
		return true
	})

	return count
}

// Implements Observer.
func (s *replaySubjectImpl[T]) IsClosed() bool {
	return s.status != KindNext
}

// Implements Observer.
func (s *replaySubjectImpl[T]) HasThrown() bool {
	return s.status == KindError
}

// Implements Observer.
func (s *replaySubjectImpl[T]) IsCompleted() bool {
	return s.status == KindComplete
}

func (s *replaySubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *replaySubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *replaySubjectImpl[T]) broadcastNext(ctx context.Context, value T) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).NextWithContext(ctx, value) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *replaySubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *replaySubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(Observer[T]).CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
