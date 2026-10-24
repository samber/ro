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
// bufferSize must be ReplaySubjectUnlimitedBufferSize (-1) or greater or equal to 0, otherwise it panics.
func NewReplaySubject[T any](bufferSize int) Subject[T] {
	if bufferSize < ReplaySubjectUnlimitedBufferSize {
		panic(ErrReplaySubjectWrongBufferSize)
	}

	return &replaySubjectImpl[T]{
		mu:         sync.Mutex{},
		terminalMu: sync.Mutex{},
		status:     KindNext,

		observers:     sync.Map{},
		observerIndex: 0,

		err:        lo.Tuple2[context.Context, error]{},
		values:     []lo.Tuple2[context.Context, T]{},
		head:       0,
		total:      0,
		bufferSize: bufferSize,
	}
}

// replayObserver is a registered observer. It only receives the values published after
// `since`, because the ones before were replayed to it on subscription.
type replayObserver[T any] struct {
	observer Observer[T]
	since    uint64
}

type replaySubjectImpl[T any] struct {
	mu     sync.Mutex // sync.RWMutex would be better, but it is too slow for high-volume subjects
	status Kind

	terminalMu sync.Mutex // serializes Error and Complete, see ErrorWithContext

	observers     sync.Map
	observerIndex uint32

	err lo.Tuple2[context.Context, error]
	// values is used as a circular buffer when bufferSize > 0: once full, the
	// oldest value lives at index head and new values overwrite it in place.
	values []lo.Tuple2[context.Context, T]
	head   int
	// total counts every value ever published, including the ones the buffer dropped.
	// The buffer holds the values numbered [total-len(values), total).
	total      uint64
	bufferSize int
}

// Implements Observable.
func (s *replaySubjectImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// Implements Observable.
func (s *replaySubjectImpl[T]) SubscribeWithContext(subscriberCtx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriber(destination)

	s.mu.Lock()

	// The lock is never held while user code runs (observer callbacks, teardowns):
	// a callback calling back into the subject would deadlock.
	//
	// The buffer is replayed outside the lock, in rounds. The observer is registered, and
	// the terminal notification delivered, only once a round finds nothing left to replay:
	// otherwise a concurrent Next could reach the observer before the values being replayed.
	var replayed uint64

	for {
		pending := s.valuesSince(replayed)
		replayed = s.total

		if len(pending) == 0 {
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

			break
		}

		s.mu.Unlock()

		for _, v := range pending {
			subscription.NextWithContext(v.A, v.B)
		}

		s.mu.Lock()

		if subscription.IsClosed() {
			// Unsubscribed from its own callback: nothing to register.
			s.mu.Unlock()
			return subscription
		}
	}

	index := atomic.AddUint32(&s.observerIndex, 1) - 1
	s.observers.Store(index, replayObserver[T]{observer: subscription, since: s.total})
	s.mu.Unlock()

	subscription.Add(func() {
		s.observers.Delete(index)
	})

	return subscription
}

// valuesSince returns, oldest first, the buffered values numbered from `from` on.
// The caller must hold s.mu.
func (s *replaySubjectImpl[T]) valuesSince(from uint64) []lo.Tuple2[context.Context, T] {
	buffered := uint64(len(s.values))
	oldest := s.total - buffered

	if from < oldest {
		from = oldest
	}

	count := s.total - from
	if count == 0 {
		return nil
	}

	// Once the buffer is full, the oldest value lives at index head.
	pending := make([]lo.Tuple2[context.Context, T], 0, count)
	for number := from; number < s.total; number++ {
		pending = append(pending, s.values[(uint64(s.head)+number-oldest)%buffered])
	}

	return pending
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
	s.mu.Lock()

	var (
		version    uint64
		dropped    T
		hasDropped bool
	)

	open := s.status == KindNext
	if open {
		s.total++
		version = s.total

		switch {
		case s.bufferSize == ReplaySubjectUnlimitedBufferSize:
			s.values = append(s.values, lo.T2(ctx, value))
		case s.bufferSize == 0:
			// The buffer cannot hold anything: the incoming value is dropped immediately.
			dropped, hasDropped = value, true
		case s.bufferSize > 0:
			if len(s.values) < s.bufferSize {
				s.values = append(s.values, lo.T2(ctx, value))
			} else {
				// Buffer is full: overwrite the oldest value in place.
				dropped, hasDropped = s.values[s.head].B, true
				s.values[s.head] = lo.T2(ctx, value)
				s.head = (s.head + 1) % s.bufferSize
			}
		}
	}

	s.mu.Unlock()

	// Broadcast outside the lock, so an observer can call back into the subject.
	// Concurrent Next calls are not ordered between them once past the lock.
	if !open {
		OnDroppedNotification(ctx, NewNotificationNext(value))
		return
	}

	s.broadcastNext(ctx, value, version)

	if hasDropped {
		OnDroppedNotification(ctx, NewNotificationNext(dropped))
	}
}

// Implements Observer.
func (s *replaySubjectImpl[T]) Error(err error) {
	s.ErrorWithContext(context.Background(), err)
}

// Implements Observer.
func (s *replaySubjectImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	// Terminal calls are serialized: a concurrent caller returns only once the first one has
	// notified every observer. Observers are released after the notification, never before,
	// or the first caller would skip the ones the second removed.
	s.terminalMu.Lock()
	defer s.terminalMu.Unlock()

	s.mu.Lock()

	open := s.status == KindNext
	if open {
		s.err = lo.T2(ctx, err)
		s.status = KindError
	}

	s.mu.Unlock()

	if open {
		s.broadcastError(ctx, err)
		s.unsubscribeAll()
	} else {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
	}
}

// Implements Observer.
func (s *replaySubjectImpl[T]) Complete() {
	s.CompleteWithContext(context.Background())
}

// Implements Observer.
func (s *replaySubjectImpl[T]) CompleteWithContext(ctx context.Context) {
	// Terminal calls are serialized: a concurrent caller returns only once the first one has
	// notified every observer. Observers are released after the notification, never before,
	// or the first caller would skip the ones the second removed.
	s.terminalMu.Lock()
	defer s.terminalMu.Unlock()

	s.mu.Lock()

	open := s.status == KindNext
	if open {
		s.status = KindComplete
	}

	s.mu.Unlock()

	if open {
		s.broadcastComplete(ctx)
		s.unsubscribeAll()
	} else {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
	}
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
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status != KindNext
}

// Implements Observer.
func (s *replaySubjectImpl[T]) HasThrown() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindError
}

// Implements Observer.
func (s *replaySubjectImpl[T]) IsCompleted() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.status == KindComplete
}

func (s *replaySubjectImpl[T]) AsObservable() Observable[T] {
	return s
}

func (s *replaySubjectImpl[T]) AsObserver() Observer[T] {
	return s
}

func (s *replaySubjectImpl[T]) broadcastNext(ctx context.Context, value T, version uint64) {
	s.observers.Range(func(_, observer any) bool {
		o := observer.(replayObserver[T]) //nolint:errcheck,forcetypeassert

		// An observer registered after this value was published got it from the replay.
		if o.since < version {
			o.observer.NextWithContext(ctx, value)
		}

		return true
	})
}

func (s *replaySubjectImpl[T]) broadcastError(ctx context.Context, err error) {
	s.observers.Range(func(_, observer any) bool {
		observer.(replayObserver[T]).observer.ErrorWithContext(ctx, err) //nolint:errcheck,forcetypeassert
		return true
	})
}

func (s *replaySubjectImpl[T]) broadcastComplete(ctx context.Context) {
	s.observers.Range(func(_, observer any) bool {
		observer.(replayObserver[T]).observer.CompleteWithContext(ctx) //nolint:errcheck,forcetypeassert
		return true
	})
}
