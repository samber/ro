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
		mu:     sync.Mutex{},
		queue:  notificationQueue{},
		status: KindNext,

		observers:     sync.Map{},
		observerIndex: 0,

		last:    lo.T2(context.TODO(), initial),
		version: 0,
		err:     lo.Tuple2[context.Context, error]{},
	}
}

// behaviorObserver is a registered observer. It only receives the values published after
// `since`, because the ones before were replayed to it on subscription.
type behaviorObserver[T any] struct {
	observer Observer[T]
	since    uint64
}

type behaviorSubjectImpl[T any] struct {
	mu     sync.Mutex // sync.RWMutex would be better, but it is too slow for high-volume subjects
	status Kind

	// queue delivers notifications outside mu, in order, and queues re-entrant calls.
	queue notificationQueue

	observers     sync.Map
	observerIndex uint32

	last lo.Tuple2[context.Context, T]
	// version is incremented on every Next. A new subscriber compares it before and after
	// replaying `last` outside the lock, to detect a value published meanwhile.
	version uint64
	err     lo.Tuple2[context.Context, error]
}

// Implements Observable.
func (s *behaviorSubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *behaviorSubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriber(destination)

	s.mu.Lock()

	// The lock is never held while user code runs (observer callbacks, teardowns):
	// a callback calling back into the subject would deadlock.
	//
	// The current value is replayed outside the lock. The observer is registered only
	// once no value has been published since the replay started, otherwise a concurrent
	// Next could reach it before the value being replayed.
	for {
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

		last, version := s.last, s.version
		s.mu.Unlock()

		// until we get a first value, should we send subscriberCtx or last.A (== context.TODO()) ?
		subscription.NextWithContext(last.A, last.B)

		s.mu.Lock()

		if subscription.IsClosed() {
			// Unsubscribed from its own callback: nothing to register.
			s.mu.Unlock()
			return subscription
		}

		if s.status == KindNext && s.version == version {
			break
		}
	}

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, behaviorObserver[T]{observer: subscription, since: s.version})
	s.mu.Unlock()

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
	s.queue.run(&s.mu, func() func() {
		if s.status != KindNext {
			return func() { OnDroppedNotification(ctx, NewNotificationNext(value)) }
		}

		s.last = lo.T2(ctx, value)
		s.version++
		version := s.version

		return func() { s.broadcastNext(ctx, value, version) }
	})
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
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
func (s *behaviorSubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	s.queue.run(&s.mu, func() func() {
		if s.status != KindNext {
			return func() { OnDroppedNotification(ctx, NewNotificationComplete[T]()) }
		}

		s.status = KindComplete

		return func() {
			s.broadcastComplete(ctx)
			s.unsubscribeAll()
		}
	})
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
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status != KindNext
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) HasThrown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindError
}

// Implements Observer.
func (s *behaviorSubjectImpl[T]) IsCompleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindComplete
}

func (s *behaviorSubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *behaviorSubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *behaviorSubjectImpl[T]) broadcastNext(ctx context.Context, value T, version uint64) {
	s.observers.Range(func(_, observer any) bool {
		o := observer.(behaviorObserver[T]) //nolint:errcheck,forcetypeassert

		// An observer registered after this value was published got it as its replayed value.
		if o.since < version {
			o.observer.NextWithContext(ctx, value)
		}

		return true
	})
}

func (s *behaviorSubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(behaviorObserver[T]).observer.ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *behaviorSubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(behaviorObserver[T]).observer.CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
