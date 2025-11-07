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
)

// Backpressure is a type that represents the backpressure strategy to use.
type Backpressure int8

const (
	// BackpressureBlock blocks the source observable when the destination is not ready to receive more values.
	BackpressureBlock Backpressure = iota
	// BackpressureDrop drops the source observable when the destination is not ready to receive more values.
	BackpressureDrop
)

// ConcurrencyMode is a type that represents the concurrency mode to use.
type ConcurrencyMode int8

// ConcurrencyModeSafe is a concurrency mode that is safe to use.
// Spinlock is ignored because it is too slow when chaining operators. Spinlock should be used
// only for short-lived local locks.
//
// ConcurrencyModeSingleProducer is an optimization targeted at hot, single-
// producer pipelines. It avoids mutex acquisition and relies on atomic state
// transitions, making it the fastest path for sequential emission from a single
// goroutine. IMPORTANT: this mode is UNSAFE if multiple goroutines may emit
// concurrently into the same subscriber. Do NOT use this mode with multi-
// producer operators such as `Merge`, `CombineLatest`, `WithConcurrency` or
// any operator that may call into a downstream subscriber from multiple
// goroutines. Use `ConcurrencyModeSafe` or `ConcurrencyModeEventuallySafe`
// when upstream concurrency is possible.
const (
	ConcurrencyModeSafe ConcurrencyMode = iota
	ConcurrencyModeUnsafe
	ConcurrencyModeEventuallySafe
	ConcurrencyModeSingleProducer
)

// Observable is the producer of values. It is the source of values that are
// emitted to Observers.
// Observable is a representation of any set of values over any amount of time.
//
// The primary method of an Observable is subscribe, which is used to attach an
// Observer to the Observable. Once an Observer is subscribed, the Observable
// may begin to emit items to the Observer. An Observable may emit any number
// of items (including zero items), then may either complete or error, but not
// both. Upon completion or error, the Observable will not emit any more items.
//
// An Observable may call an Observer's methods synchronously or asynchronously.
//
// An Observable is not a stream. It is a factory for streams.
type Observable[T any] interface {
	// Subscribe subscribes an Observer to the Observable. The Observer will begin
	// to receive items emitted by the Observable. The Observer may receive any
	// number of items (including zero items), then may either complete or error,
	// but not both. Upon completion or error, the Observer will not receive any
	// more items.
	//
	// The Subscribe method returns a Subscription that can be used to unsubscribe
	// the Observer from the Observable. The Subscription may be used to cancel the
	// subscription, and to wait for the subscription to complete.
	//
	// The Subscription might be already disposed when the Subscribe method returns.
	// In this case, the Teardown function is not called.
	//
	// The Subscribe method may call the Observer's methods synchronously or
	// asynchronously. The Observer is responsible for handling concurrency and
	// synchronization.
	Subscribe(destination Observer[T]) Subscription
	SubscribeWithContext(ctx context.Context, destination Observer[T]) Subscription
}

var _ Observable[int] = (*observableImpl[int])(nil)

// NewObservable creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewObservable[T any](subscribe func(destination Observer[T]) Teardown) Observable[T] {
	return NewSafeObservable(subscribe)
}

// NewSafeObservable creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewSafeObservable[T any](subscribe func(destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(
		func(ctx context.Context, destination Observer[T]) Teardown {
			return subscribe(destination)
		},
		ConcurrencyModeSafe,
	)
}

// NewUnsafeObservable creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewUnsafeObservable[T any](subscribe func(destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(
		func(ctx context.Context, destination Observer[T]) Teardown {
			return subscribe(destination)
		},
		ConcurrencyModeUnsafe,
	)
}

// NewEventuallySafeObservable creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is safe for concurrent use, but concurrent messages are dropped.
func NewEventuallySafeObservable[T any](subscribe func(destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(
		func(ctx context.Context, destination Observer[T]) Teardown {
			return subscribe(destination)
		},
		ConcurrencyModeEventuallySafe,
	)
}

// NewSingleProducerObservable creates a new Observable optimized for single producer scenarios.
// The subscribe function is called when the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both. Upon completion or error,
// the Observable will not emit any more items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewSingleProducerObservable[T any](subscribe func(destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(
		func(ctx context.Context, destination Observer[T]) Teardown {
			return subscribe(destination)
		},
		ConcurrencyModeSingleProducer,
	)
}

// NewObservableWithContext creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) Observable[T] {
	return NewSafeObservableWithContext(subscribe)
}

// NewSafeObservableWithContext creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewSafeObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(subscribe, ConcurrencyModeSafe)
}

// NewUnsafeObservableWithContext creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewUnsafeObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(subscribe, ConcurrencyModeUnsafe)
}

// NewEventuallySafeObservableWithContext creates a new Observable. The subscribe function is called when
// the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error,
// but not both. Upon completion or error, the Observable will not emit any more
// items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is safe for concurrent use, but concurrent messages are dropped.
func NewEventuallySafeObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(subscribe, ConcurrencyModeEventuallySafe)
}

// NewSingleProducerObservableWithContext creates a new Observable optimized for single producer scenarios.
// The subscribe function is called when the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both. Upon completion or error, the Observable will not emit any more items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// This method is not safe for concurrent use.
func NewSingleProducerObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) Observable[T] {
	return NewObservableWithConcurrencyMode(subscribe, ConcurrencyModeSingleProducer)
}

// NewObservableWithConcurrencyMode creates a new Observable with the given concurrency mode.
// The subscribe function is called when the Observable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both. Upon completion or error, the Observable will not emit any more items.
//
// The subscribe function should return a Teardown function that will be called
// when the Subscription is unsubscribed. The Teardown function should clean up
// any resources created during the subscription.
//
// The subscribe function may return a Teardown function that does nothing, if
// no cleanup is necessary. In this case, the Teardown function should return nil.
//
// The Observable will use the given concurrency mode.
//
// It is rarely used as a public API.
func NewObservableWithConcurrencyMode[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown, mode ConcurrencyMode) Observable[T] {
	return &observableImpl[T]{
		mode:      mode,
		subscribe: subscribe,
	}
}

type observableImpl[T any] struct {
	mode      ConcurrencyMode
	subscribe func(ctx context.Context, destination Observer[T]) Teardown
}

// Subscribe subscribes an Observer to the Observable. The Observer will begin
// to receive items emitted by the Observable. The Observer may receive any
// number of items (including zero items), then may either complete or error,
// but not both. Upon completion or error, the Observer will not receive any
// more items.
//
// The Subscribe method returns a Subscription that can be used to unsubscribe
// the Observer from the Observable. The Subscription may be used to cancel the
// subscription, and to wait for the subscription to complete.
//
// The Subscription might be already disposed when the Subscribe method returns.
// In this case, the Teardown function is not called.
//
// The Subscribe method may call the Observer's methods synchronously or
// asynchronously. The Observer is responsible for handling concurrency and
// synchronization.
func (s *observableImpl[T]) Subscribe(destination Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), destination)
}

// SubscribeWithContext subscribes an Observer to the Observable. The Observer will begin
// to receive items emitted by the Observable. The Observer may receive any
// number of items (including zero items), then may either complete or error,
// but not both. Upon completion or error, the Observer will not receive any
// more items.
//
// The Subscribe method returns a Subscription that can be used to unsubscribe
// the Observer from the Observable. The Subscription may be used to cancel the
// subscription, and to wait for the subscription to complete.
//
// The Subscription might be already disposed when the Subscribe method returns.
// In this case, the Teardown function is not called.
//
// The Subscribe method may call the Observer's methods synchronously or
// asynchronously. The Observer is responsible for handling concurrency and
// synchronization.
func (s *observableImpl[T]) SubscribeWithContext(ctx context.Context, destination Observer[T]) Subscription {
	subscription := NewSubscriberWithConcurrencyMode(destination, s.mode)
	// Compute effective panic-capture policy once at subscription time and
	// configure the subscriber hot-path helpers to avoid per-notification
	// context lookups. If the destination is not an internal observerImpl,
	// we conservatively assume capture is enabled.
	capture := true
	if oi, ok := destination.(*observerImpl[T]); ok {
		capture = oi.capturePanics && !isObserverPanicCaptureDisabled(ctx)
	} else if isObserverPanicCaptureDisabled(ctx) {
		// For external observer implementations, respect context opt-out.
		capture = false
	}

	// If subscription is our concrete subscriberImpl, set direct call helpers.
	if ssub, ok := subscription.(*subscriberImpl[T]); ok {
		// Avoid configuring directors when NewSubscriber returned the input
		// subscriber itself (subscription == destination) — in that case the
		// destination is already a Subscriber and it is responsible for its
		// own hot-path wiring.
		if subscription != destination {
			ssub.setDirectors(destination, capture)
		}
	}

	// If panic capture is explicitly disabled on the subscription context and
	// the observable is in an unsafe/single-producer mode, skip the TryCatch
	// wrapper to avoid extra allocations on the subscribe path. Callers should
	// use `WithObserverPanicCaptureDisabled(ctx)` when subscribing in
	// performance-sensitive code.
	if isObserverPanicCaptureDisabled(ctx) && (s.mode == ConcurrencyModeUnsafe || s.mode == ConcurrencyModeSingleProducer) {
		subscription.Add(s.subscribe(ctx, subscription))
		return subscription
	}

	func() {
		defer func() {
			if e := recover(); e != nil {
				err := recoverValueToError(e)
				subscription.ErrorWithContext(ctx, newObservableError(err))
				subscription.Unsubscribe()
			}
		}()

		// Warning: here, we are catching panic in subscription.Add.
		// I'm not sure if it's a good idea.
		subscription.Add(s.subscribe(ctx, subscription))
	}()

	return subscription
}

// Collect collects all values emitted by the source Observable and returns them
// as a slice. It waits for the source Observable to complete before returning.
// If the source Observable emits an error, the error is returned along with the
// values collected so far.
func Collect[T any](obs Observable[T]) ([]T, error) {
	v, _, err := CollectWithContext(context.Background(), obs)
	return v, err
}

// CollectWithContext collects all values emitted by the source Observable and returns them
// as a slice. It waits for the source Observable to complete before returning.
// If the source Observable emits an error, the error is returned along with the
// values collected so far.
// @TODO: return more values, such as (isCanceled bool) or (duration time.Duration) ?
func CollectWithContext[T any](ctx context.Context, obs Observable[T]) ([]T, context.Context, error) {
	values := []T{}

	var lastCtx context.Context
	var err error

	sub := obs.SubscribeWithContext(
		ctx,
		NewObserverWithContext(
			func(ctx context.Context, value T) {
				values = append(values, value)
			},
			func(ctx context.Context, thrown error) {
				err = thrown
				lastCtx = ctx
			},
			func(ctx context.Context) {
				lastCtx = ctx
			},
		),
	)

	sub.Wait() // Note: using .Wait() is not recommended.

	return values, lastCtx, err
}

// ConnectableObservable is an Observable that can be connected and disconnected.
// When connected, it will emit values to its observers.
//
// ConnectableObservable is useful when you want to share a single subscription to an Observable
// among multiple observers. This is useful when you want to multicast the values of an Observable.
type ConnectableObservable[T any] interface {
	Observable[T]

	// Connect connects the ConnectableObservable. When connected, the ConnectableObservable
	// will emit values to its observers. If the ConnectableObservable is already connected,
	// this method creates a new subscription and starts emitting values to its observers.
	//
	// The Connect method returns a Subscription that can be used to disconnect the
	// ConnectableObservable. The Subscription may be used to cancel the connection,
	// and to wait for the connection to complete.
	//
	// The Subscription might be already disposed when the Connect method returns.
	Connect() Subscription
	ConnectWithContext(ctx context.Context) Subscription
}

var (
	_ ConnectableObservable[int] = (*connectableObservableImpl[int])(nil)
	_ Observable[int]            = (*connectableObservableImpl[int])(nil)
)

// ConnectableConfig is the configuration for a ConnectableObservable.
type ConnectableConfig[T any] struct {
	Connector         func() Subject[T]
	ResetOnDisconnect bool
}

func defaultConnector[T any]() Subject[T] {
	return NewPublishSubject[T]()
}

// NewConnectableObservable creates a new ConnectableObservable. The subscribe function is called when
// the ConnectableObservable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both.
// Upon completion or error, the ConnectableObservable will not emit any more items.
//
// The ConnectableObservable will use the default connector, which is a PublishSubject.
// The ConnectableObservable will reset the source when disconnected. This means that
// when the ConnectableObservable is disconnected, it will create a new source when
// reconnected.
//
// If you want to use a different connector or change the reset behavior, use
// NewConnectableObservableWithConfig.
func NewConnectableObservable[T any](subscribe func(destination Observer[T]) Teardown) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		NewObservable(subscribe),
		ConnectableConfig[T]{
			Connector:         defaultConnector[T],
			ResetOnDisconnect: true,
		},
	)
}

// NewConnectableObservableWithContext creates a new ConnectableObservable. The subscribe function is called when
// the ConnectableObservable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both.
// Upon completion or error, the ConnectableObservable will not emit any more items.
//
// The ConnectableObservable will use the default connector, which is a PublishSubject.
// The ConnectableObservable will reset the source when disconnected. This means that
// when the ConnectableObservable is disconnected, it will create a new source when
// reconnected.
//
// If you want to use a different connector or change the reset behavior, use
// NewConnectableObservableWithConfig.
func NewConnectableObservableWithContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		NewObservableWithContext(subscribe),
		ConnectableConfig[T]{
			Connector:         defaultConnector[T],
			ResetOnDisconnect: true,
		},
	)
}

// NewConnectableObservableWithConfig creates a new ConnectableObservable. The subscribe function is called when
// the ConnectableObservable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both.
// Upon completion or error, the ConnectableObservable will not emit any more items.
//
// The ConnectableObservable will use the given connector. The ConnectableObservable will reset
// the source when disconnected if ResetOnDisconnect is true. This means that when the
// ConnectableObservable is disconnected, it will create a new source when reconnected.
func NewConnectableObservableWithConfig[T any](subscribe func(destination Observer[T]) Teardown, config ConnectableConfig[T]) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		NewObservable(subscribe),
		config,
	)
}

// NewConnectableObservableWithConfigAndContext creates a new ConnectableObservable. The subscribe function is called when
// the ConnectableObservable is subscribed to. The subscribe function is given an Observer,
// to which it may emit any number of items, then may either complete or error, but not both.
// Upon completion or error, the ConnectableObservable will not emit any more items.
//
// The ConnectableObservable will use the given connector. The ConnectableObservable will reset
// the source when disconnected if ResetOnDisconnect is true. This means that when the
// ConnectableObservable is disconnected, it will create a new source when reconnected.
func NewConnectableObservableWithConfigAndContext[T any](subscribe func(ctx context.Context, destination Observer[T]) Teardown, config ConnectableConfig[T]) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		NewObservableWithContext(subscribe),
		config,
	)
}

// Connectable creates a new ConnectableObservable from an Observable. The ConnectableObservable
// will use the default connector, which is a PublishSubject. The ConnectableObservable will reset
// the source when disconnected. This means that when the ConnectableObservable is disconnected,
// it will create a new source when reconnected.
//
// If you want to use a different connector or change the reset behavior, use ConnectableWithConfig.
func Connectable[T any](source Observable[T]) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		source,
		ConnectableConfig[T]{
			Connector:         defaultConnector[T],
			ResetOnDisconnect: true,
		},
	)
}

// ConnectableWithConfig creates a new ConnectableObservable from an Observable. The ConnectableObservable
// will use the given connector. The ConnectableObservable will reset the source when disconnected
// if ResetOnDisconnect is true. This means that when the ConnectableObservable is disconnected,
// it will create a new source when reconnected.
func ConnectableWithConfig[T any](source Observable[T], config ConnectableConfig[T]) ConnectableObservable[T] {
	return newConnectableObservableImpl(
		source,
		config,
	)
}

func newConnectableObservableImpl[T any](source Observable[T], config ConnectableConfig[T]) ConnectableObservable[T] {
	if config.Connector == nil {
		panic(ErrConnectableObservableMissingConnectorFactory)
	}

	return &connectableObservableImpl[T]{
		config:       config,
		source:       source,
		subject:      config.Connector(),
		subscription: nil,
	}
}

// isBackgroundContext detects whether the provided context is a top-level
// background or TODO context. We compare pointers to the well-known
// background/TODO values so callers can decide whether the provided context
// is the default empty context. This is intentionally conservative — many
// derived contexts will not be equal to Background/TODO and therefore will
// be treated as explicit contexts.
func isBackgroundContext(ctx context.Context) bool {
	return ctx == context.Background() || ctx == context.TODO()
}

type connectableObservableImpl[T any] struct {
	mu           sync.Mutex
	config       ConnectableConfig[T]
	source       Observable[T]
	subject      Subject[T]
	subscription Subscription
	// lastSubscriberCtx stores the most-recent non-nil context passed to
	// SubscribeWithContext. It is used as a fallback when Connect() is
	// called without an explicit context so that the source subscription
	// inherits the subscriber's context (values, cancellation) as tests
	// expect.
	lastSubscriberCtx context.Context
}

// Connect connects the ConnectableObservable. When connected, the ConnectableObservable
// will emit values to its observers. If the ConnectableObservable is already connected,
// this method creates a new subscription and starts emitting values to its observers.
//
// The Connect method returns a Subscription that can be used to disconnect the
// ConnectableObservable. The Subscription may be used to cancel the connection,
// and to wait for the connection to complete.
//
// The Subscription might be already disposed when the Connect method returns.
func (s *connectableObservableImpl[T]) Connect() Subscription {
	return s.ConnectWithContext(context.Background())
}

// ConnectWithContext connects the ConnectableObservable. When connected, the ConnectableObservable
// will emit values to its observers. If the ConnectableObservable is already connected,
// this method creates a new subscription and starts emitting values to its observers.
//
// The Connect method returns a Subscription that can be used to disconnect the
// ConnectableObservable. The Subscription may be used to cancel the connection,
// and to wait for the connection to complete.
//
// The Subscription might be already disposed when the Connect method returns.
func (s *connectableObservableImpl[T]) ConnectWithContext(ctx context.Context) Subscription {
	s.mu.Lock()
	if s.subscription == nil || s.subscription.IsClosed() {
		// If caller passed a background context and we have a subscriber
		// context stored, prefer that so subscribers' context values are
		// visible to the source. This mirrors expected behavior in tests
		// where SubscribeWithContext is called before Connect().
		effectiveCtx := ctx
		if isBackgroundContext(ctx) && s.lastSubscriberCtx != nil {
			effectiveCtx = s.lastSubscriberCtx
		}

		s.subscription = s.source.SubscribeWithContext(effectiveCtx, s.subject)
		s.mu.Unlock()
		s.subscription.Add(func() {
			if s.config.ResetOnDisconnect {
				s.subject = s.config.Connector()
			}
		})
	} else {
		s.mu.Unlock()
	}

	return s.subscription
}

func (s *connectableObservableImpl[T]) Subscribe(observer Observer[T]) Subscription {
	return s.SubscribeWithContext(context.Background(), observer)
}

func (s *connectableObservableImpl[T]) SubscribeWithContext(ctx context.Context, observer Observer[T]) Subscription {
	// Record the subscriber context so Connect() can use it if caller
	// didn't provide an explicit context. We store the most recent
	// non-background context.
	if ctx != nil && !isBackgroundContext(ctx) {
		s.mu.Lock()
		s.lastSubscriberCtx = ctx
		s.mu.Unlock()
	}

	return s.subject.SubscribeWithContext(ctx, observer)
}
