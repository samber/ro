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

	"github.com/samber/lo"
)

// UnicastSubjectUnlimitedBufferSize is the unlimited buffer size for a UnicastSubject.
const UnicastSubjectUnlimitedBufferSize = -1

var _ Subject[int] = (*unicastSubjectImpl[int])(nil)

// NewUnicastSubject queues up events until a single Observer subscribes to it,
// replays those events to it until the Observer catches up and then switches
// to relaying events live to this single Observer.
// bufferSize must be UnicastSubjectUnlimitedBufferSize (-1) or greater or equal to 0, otherwise it panics.
func NewUnicastSubject[T any](bufferSize int) Subject[T] {
	if bufferSize < UnicastSubjectUnlimitedBufferSize {
		panic(ErrUnicastSubjectWrongBufferSize)
	}

	return &unicastSubjectImpl[T]{
		mu:     sync.Mutex{},
		status: KindNext,

		observer:  nil,
		replaying: false,

		err:        lo.Tuple2[context.Context, error]{},
		values:     []lo.Tuple2[context.Context, T]{},
		bufferSize: bufferSize,
	}
}

type unicastSubjectImpl[T any] struct {
	mu     sync.Mutex // sync.RWMutex would be better, but it is too slow for high-volume subjects
	status Kind

	observer  Observer[T]
	replaying bool // true while the single observer drains the buffer, outside the lock

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
		s.mu.Unlock()

		subscription.CompleteWithContext(subscriberCtx)

		return subscription
	}

	if s.observer != nil || s.replaying {
		s.mu.Unlock()

		subscription.ErrorWithContext(subscriberCtx, ErrUnicastSubjectConcurrent)

		return subscription
	}

	s.replaying = true
	s.mu.Unlock()

	// Drain the buffer outside the lock, so the observer may call back into the
	// subject or unsubscribe. Live notifications arriving meanwhile are appended to
	// s.values (observer is still nil) and picked up by the next iteration, which
	// preserves order. The observer is published only once the buffer is empty,
	// in the same critical section that reads the terminal status.
	for {
		s.mu.Lock()

		if len(s.values) == 0 || subscription.IsClosed() {
			s.replaying = false

			status, errCtx, err := s.status, s.err.A, s.err.B
			if status == KindNext && !subscription.IsClosed() {
				s.observer = subscription
			}

			s.mu.Unlock()

			// A terminal notification that arrived during the replay found no observer.
			// Deliver it now, after the buffered values.
			switch status {
			case KindError:
				subscription.ErrorWithContext(errCtx, err)
			case KindComplete:
				subscription.CompleteWithContext(subscriberCtx)
			case KindNext:
				// still live: the observer receives the next notifications directly
			}

			break
		}

		batch := s.values
		s.values = []lo.Tuple2[context.Context, T]{}
		s.mu.Unlock()

		for i, v := range batch {
			if subscription.IsClosed() {
				// Keep the undelivered values for the next subscriber.
				s.mu.Lock()
				s.values = append(append([]lo.Tuple2[context.Context, T]{}, batch[i:]...), s.values...)
				s.mu.Unlock()

				break
			}

			subscription.NextWithContext(v.A, v.B)
		}
	}

	// No lock held here: if the subscription is already closed, Add runs the
	// teardown inline and it may take s.mu safely.
	subscription.Add(func() {
		s.mu.Lock()

		if s.observer == subscription { // never clear a newer observer
			s.observer = nil
		}

		s.mu.Unlock()
	})

	return subscription
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Next(value T) {
	s.NextWithContext(context.Background(), value)
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) NextWithContext(ctx context.Context, value T) {
	s.mu.Lock()

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

	s.mu.Unlock()
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	s.mu.Lock()

	if s.status == KindNext {
		s.err = lo.T2(ctx, err)
		s.status = KindError

		if s.observer != nil {
			tmp := s.observer
			s.observer = nil

			defer tmp.ErrorWithContext(ctx, err)
		} else if !s.replaying { // during a replay, it is delivered once the buffer is drained
			OnDroppedNotification(ctx, NewNotificationError[T](err))
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
	}

	s.mu.Unlock()
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	s.mu.Lock()

	if s.status == KindNext {
		s.status = KindComplete

		if s.observer != nil {
			tmp := s.observer
			s.observer = nil

			defer tmp.CompleteWithContext(ctx)
		} else if !s.replaying { // during a replay, it is delivered once the buffer is drained
			OnDroppedNotification(ctx, NewNotificationComplete[T]())
		}
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}

	s.mu.Unlock()
}

func (s *unicastSubjectImpl[T]) HasObserver() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.observer != nil
}

func (s *unicastSubjectImpl[T]) CountObservers() int {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.observer != nil {
		return 1
	}

	return 0
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) IsClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status != KindNext
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) HasThrown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindError
}

// Implements Observer.
func (s *unicastSubjectImpl[T]) IsCompleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindComplete
}

func (s *unicastSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *unicastSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}
