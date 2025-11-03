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
	"fmt"
	"sync/atomic"

	"github.com/samber/lo"
)

// observerPanicCaptureEnabled controls whether observer callbacks are wrapped
// with panic-recovery logic (via lo.TryCatchWithErrorValue). This flag is
// intentionally unexported to avoid unsynchronized mutations from tests or
// consumers. Tests that need to toggle the global behaviour should use the
// `panicCaptureGuard` helper in `observer_test.go` which serialises writers
// and allows concurrent readers via an RWMutex. Use the exported helpers
// SetCaptureObserverPanics / CaptureObserverPanics for thread-safe access.
var observerPanicCaptureEnabled atomic.Bool

func init() {
	observerPanicCaptureEnabled.Store(true)
}

// SetCaptureObserverPanics toggles panic recovery in observer callbacks.
//
// WARNING: this modifies a global package-level setting. The flag is sampled
// when an Observer is constructed (e.g. via NewObserver / NewObserverWithContext)
// so changing it at runtime affects *newly created* Observers only — existing
// Observer instances retain the value they captured at construction time.
//
// Because this is global mutable state, prefer setting it once at application
// startup for benchmarking or performance-sensitive pipelines. Tests that need
// to toggle the value should use the coordination helpers in
// `observer_test.go` (the `panicCaptureGuard`) to avoid races.
func SetCaptureObserverPanics(enabled bool) {
	observerPanicCaptureEnabled.Store(enabled)
}

// CaptureObserverPanics reports whether observer callbacks are wrapped in panic recovery.
func CaptureObserverPanics() bool {
	return observerPanicCaptureEnabled.Load()
}

// Observer is the consumer of an Observable. It receives notifications: Next,
// Error, and Complete. Observers are safe for concurrent calls to Next,
// Error, and Complete. It is the responsibility of the Observer to ensure
// that notifications are not forwarded after it has been closed.
type Observer[T any] interface {
	// Next receives the next value from the Observable. It is called zero or
	// more times by the Observable. The Observable may call Next synchronously
	// or asynchronously. If Next is called after the Observer has been closed,
	// the value will be dropped.
	Next(value T)
	NextWithContext(ctx context.Context, value T)
	// Error receives an error from the Observable. It is called at most once by
	// the Observable. The Observable may call Error synchronously or
	// asynchronously. If Error is called after the Observer has been closed, the
	// error will be dropped.
	Error(err error)
	ErrorWithContext(ctx context.Context, err error)
	// Complete receives a completion notification from the Observable. It is called
	// at most once by the Observable. The Observable may call Complete
	// synchronously or asynchronously. If Complete is called after the Observer has
	// been closed, the completion notification will be dropped.
	Complete()
	CompleteWithContext(ctx context.Context)

	// IsClosed returns true if the Observer has been closed, either by an error
	// or completion notification. If the Observer is closed, it will not receive
	// any more notifications.
	IsClosed() bool
	// HasThrown returns true if the Observer has received an error notification.
	HasThrown() bool
	// IsCompleted returns true if the Observer has received a completion notification.
	IsCompleted() bool
}

/************************
 *     Base Observer    *
 ************************/

var _ Observer[int] = (*observerImpl[int])(nil)

// NewObserver creates a new Observer with the provided callbacks. No context
// is provided.
func NewObserver[T any](onNext func(value T), onError func(err error), onComplete func()) Observer[T] {
	return &observerImpl[T]{
		status:        0,
		capturePanics: observerPanicCaptureEnabled.Load(),
		onNext: func(ctx context.Context, value T) {
			onNext(value)
		},
		onError: func(ctx context.Context, err error) {
			onError(err)
		},
		onComplete: func(ctx context.Context) {
			onComplete()
		},
	}
}

// NewObserverWithContext creates a new Observer with the provided callbacks. A context
// is provided to each callback.
func NewObserverWithContext[T any](onNext func(ctx context.Context, value T), onError func(ctx context.Context, err error), onComplete func(ctx context.Context)) Observer[T] {
	return &observerImpl[T]{
		status:        0,
		capturePanics: observerPanicCaptureEnabled.Load(),
		onNext:        onNext,
		onError:       onError,
		onComplete:    onComplete,
	}
}

type observerImpl[T any] struct {
	// 0: active
	// 1: errored
	// 2: completed
	status        int32
	capturePanics bool
	onNext        func(context.Context, T)
	onError       func(context.Context, error) // @TODO: add a default onError that log the error ?
	onComplete    func(context.Context)
}

func (o *observerImpl[T]) Next(value T) {
	o.NextWithContext(context.Background(), value)
}

func (o *observerImpl[T]) NextWithContext(ctx context.Context, value T) {
	if o.onNext == nil || atomic.LoadInt32(&o.status) != 0 {
		OnDroppedNotification(ctx, NewNotificationNext(value))
		return
	}

	o.tryNext(ctx, value)
}

func (o *observerImpl[T]) Error(err error) {
	o.ErrorWithContext(context.Background(), err)
}

func (o *observerImpl[T]) ErrorWithContext(ctx context.Context, err error) {
	if o.onError == nil || !atomic.CompareAndSwapInt32(&o.status, 0, 1) {
		OnDroppedNotification(ctx, NewNotificationError[T](err))
		return
	}

	o.tryError(ctx, err)
}

func (o *observerImpl[T]) Complete() {
	o.CompleteWithContext(context.Background())
}

func (o *observerImpl[T]) CompleteWithContext(ctx context.Context) {
	if o.onComplete == nil || !atomic.CompareAndSwapInt32(&o.status, 0, 2) {
		OnDroppedNotification(ctx, NewNotificationComplete[T]())
		return
	}

	o.tryComplete(ctx)
}

func (o *observerImpl[T]) tryNext(ctx context.Context, value T) {
	if !o.capturePanics {
		o.onNext(ctx, value)
		return
	}

	lo.TryCatchWithErrorValue(
		func() error {
			o.onNext(ctx, value)
			return nil
		},
		func(e any) {
			err := newObserverError(recoverValueToError(e))

			if o.onError == nil {
				OnUnhandledError(ctx, err)
			} else {
				o.tryError(ctx, err)
			}
		},
	)
}

func (o *observerImpl[T]) tryError(ctx context.Context, err error) {
	if !o.capturePanics {
		o.onError(ctx, err)
		return
	}

	lo.TryCatchWithErrorValue(
		func() error {
			o.onError(ctx, err)
			return nil
		},
		func(e any) {
			err := newObserverError(recoverValueToError(e))
			OnUnhandledError(ctx, err)
		},
	)
}

func (o *observerImpl[T]) tryComplete(ctx context.Context) {
	if !o.capturePanics {
		o.onComplete(ctx)
		return
	}

	lo.TryCatchWithErrorValue(
		func() error {
			o.onComplete(ctx)
			return nil
		},
		func(e any) {
			err := newObserverError(recoverValueToError(e))
			OnUnhandledError(ctx, err)
		},
	)
}

func (o *observerImpl[T]) IsClosed() bool {
	return atomic.LoadInt32(&o.status) != 0
}

func (o *observerImpl[T]) HasThrown() bool {
	return atomic.LoadInt32(&o.status) == 1
}

func (o *observerImpl[T]) IsCompleted() bool {
	return atomic.LoadInt32(&o.status) == 2
}

/*********************
 * Partial Observers *
 *********************/

// OnNext is a partial Observer with only the Next method implemented.
// Warning: This observer will silent errors.
func OnNext[T any](onNext func(value T)) Observer[T] {
	onError := func(err error) {}
	onComplete := func() {}

	return NewObserver(onNext, onError, onComplete)
}

// OnNextWithContext is a partial Observer with only the Next method implemented.
// Warning: This observer will silent errors.
func OnNextWithContext[T any](onNext func(ctx context.Context, value T)) Observer[T] {
	onError := func(ctx context.Context, err error) {}
	onComplete := func(ctx context.Context) {}

	return NewObserverWithContext(onNext, onError, onComplete)
}

// OnError is a partial Observer with only the Error method implemented.
func OnError[T any](onError func(err error)) Observer[T] {
	onNext := func(value T) {}
	onComplete := func() {}

	return NewObserver(onNext, onError, onComplete)
}

// OnErrorWithContext is a partial Observer with only the Error method implemented.
func OnErrorWithContext[T any](onError func(ctx context.Context, err error)) Observer[T] {
	onNext := func(ctx context.Context, value T) {}
	onComplete := func(ctx context.Context) {}

	return NewObserverWithContext(onNext, onError, onComplete)
}

// OnComplete is a partial Observer with only the Complete method implemented.
// Warning: This observer will silent errors.
func OnComplete[T any](onComplete func()) Observer[T] {
	onNext := func(value T) {}
	onError := func(err error) {}

	return NewObserver(onNext, onError, onComplete)
}

// OnCompleteWithContext is a partial Observer with only the Complete method implemented.
// Warning: This observer will silent errors.
func OnCompleteWithContext[T any](onComplete func(ctx context.Context)) Observer[T] {
	onNext := func(ctx context.Context, value T) {}
	onError := func(ctx context.Context, err error) {}

	return NewObserverWithContext(onNext, onError, onComplete)
}

// NoopObserver is an Observer that does nothing.
// Warning: This observer will silent errors.
func NoopObserver[T any]() Observer[T] {
	return NewObserverWithContext(
		func(ctx context.Context, value T) {},
		func(ctx context.Context, err error) {},
		func(ctx context.Context) {},
	)
}

// PrintObserver is an utilitary Observer that dump notifications for debug purpose.
func PrintObserver[T any]() Observer[T] {
	return NewObserverWithContext(
		func(ctx context.Context, value T) {
			fmt.Printf("Next: %v\n", value)
		},
		func(ctx context.Context, err error) {
			fmt.Printf("Error: %s\n", err.Error())
		},
		func(ctx context.Context) {
			fmt.Printf("Completed\n")
		},
	)
}
