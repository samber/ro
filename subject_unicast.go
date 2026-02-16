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

	"github.com/samber/lo"
)

// UnicastSubjectUnlimitedBufferSize is the unlimited buffer size for a UnicastSubject.
const UnicastSubjectUnlimitedBufferSize = -1

var _ Subject[int] = (*unicastSubjectImpl[int])(nil)

// NewUnicastSubject queues up events until a single Observer subscribes to it,
// replays those events to it until the Observer catches up and then switches
// to relaying events live to this single Observer.
func NewUnicastSubject[T any](bufferSize int) Subject[T] {
	return &unicastSubjectImpl[T]{
		status: KindNext,

		observer: nil,

		err:        lo.Tuple2[context.Context, error]{},
		values:     []lo.Tuple2[context.Context, T]{},
		bufferSize: bufferSize,
	}
}

type unicastSubjectImpl[T any] struct {
	status Kind

	observer Observer[T]

	err        lo.Tuple2[context.Context, error]
	values     []lo.Tuple2[context.Context, T]
	bufferSize int
}

// Implements Observable.
func (s *unicastSubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *unicastSubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
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

	if s.observer != nil {
		subscription.ErrorWithContext(subscriberCtx, ErrUnicastSubjectConcurrent)
		return subscription
	}

	for _, v := range s.values {
		subscription.NextWithContext(v.A, v.B)
	}

	s.values = []lo.Tuple2[context.Context, T]{}

	s.observer = subscription

	subscription.Add(func() {
		s.observer = nil
	})

	return subscription
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	if s.status == KindNext { //nolint:nestif
		if s.observer != nil {
			tmp := s.observer
			defer tmp.NextWithContext(ctx, value) // out of lock
		} else {
			s.values = append(s.values, lo.T2(ctx, value))
			if s.bufferSize != UnicastSubjectUnlimitedBufferSize && len(s.values) > s.bufferSize {
				OnDroppedNotification(ctx, NewNotificationNext(s.values[0].B))
				s.values = s.values[len(s.values)-s.bufferSize:]
			}
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationNext(value))
	}
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	if s.status == KindNext {
		s.err = lo.T2(ctx, err)
		s.status = KindError

		if s.observer != nil {
			tmp := s.observer
			s.observer = nil

			defer tmp.ErrorWithContext(ctx, err)
		} else {
			OnDroppedNotification(ctx, NewNotificationError[T](err))
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
	}
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	if s.status == KindNext {
		s.status = KindComplete

		if s.observer != nil {
			tmp := s.observer
			s.observer = nil

			defer tmp.CompleteWithContext(ctx)
		} else {
			OnDroppedNotification(ctx, NewNotificationComplete[T]())
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}
}

func (s *unicastSubjectImpl[T]) HasObserver() bool {
	return s.observer != nil
}

func (s *unicastSubjectImpl[T]) CountObservers() int {
	if s.observer != nil {
		return 1
	}

	return 0
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) IsClosed() bool {
	return s.status != KindNext
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) HasThrown() bool {
	return s.status == KindError
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) IsCompleted() bool {
	return s.status == KindComplete
}

func (s *unicastSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *unicastSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}
