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
	"github.com/samber/ro/internal/xerrors"
)

// Teardown is a function that cleans up resources, such as closing
// a file or a network connection. It is called when the Subscription is closed.
// It is part of a Subscription, and is returned by the Observable creation.
// It will be called only once, when the Subscription is canceled.
type Teardown func()
type TeardownWithContext func(ctx context.Context)

// Unsubscribable represents any type that can be unsubscribed from.
// It provides a common interface for cancellation operations.
type Unsubscribable interface {
	Unsubscribe()
	UnsubscribeWithContext(ctx context.Context)
}

// Subscription represents an ongoing execution of an `Observable`, and has
// a minimal API which allows you to cancel that execution.
type Subscription interface {
	Unsubscribable

	Add(teardown Teardown)
	AddWithContext(teardown TeardownWithContext)
	AddUnsubscribable(unsubscribable Unsubscribable)
	IsClosed() bool
	Wait() // Note: using .Wait() is not recommended.
}

type subscriptionImpl struct {
    done          bool
    mu            sync.Mutex
    finalizers    []Teardown
    ctxFinalizers []TeardownWithContext

}

var _ Subscription = (*subscriptionImpl)(nil)

// NewSubscription creates a new Subscription. When `teardown` is nil, nothing
// is added. When the subscription is already disposed, the `teardown` callback
// is triggered immediately.
func NewSubscription(teardown Teardown) Subscription {
	s := &subscriptionImpl{
		finalizers: []Teardown{},
		ctxFinalizers: []TeardownWithContext{},
	}
	if teardown != nil {
		s.finalizers = append(s.finalizers, teardown)
	}

	return s
}

func NewSubscriptionWithContext(teardown TeardownWithContext) Subscription {
	s := &subscriptionImpl{
		finalizers:    []Teardown{},
		ctxFinalizers: []TeardownWithContext{},

	}

	if teardown != nil {
		s.ctxFinalizers = append(s.ctxFinalizers, teardown)
	}

	

	return s
}

// Add receives a finalizer to execute upon unsubscription. When `teardown`
// is nil, nothing is added. When the subscription is already disposed, the `teardown`
// callback is triggered immediately.
//
// This method is thread-safe.
//
// Implements Subscription.
func (s *subscriptionImpl) Add(teardown Teardown) {
	if teardown == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.done {
		_ = execFinalizer(teardown)
		return
	} 

	s.finalizers = append(s.finalizers, teardown)
}

// AddWithContext registers a teardown function that receives a context when
// the subscription is unsubscribed.
func (s *subscriptionImpl) AddWithContext(teardown TeardownWithContext) {
	if teardown == nil {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.done {
		_ = execFinalizerWithContext(teardown, context.Background())
		return
	}

	s.ctxFinalizers = append(s.ctxFinalizers, teardown)
}

// AddUnsubscribable merges multiple subscriptions into one. The method does nothing
// if `unsubscribable` is nil.
//
// This method is thread-safe.
//
// Implements Subscription.
func (s *subscriptionImpl) AddUnsubscribable(unsubscribable Unsubscribable) {
	if unsubscribable == nil {
		return
	}

	s.Add(func() {
		unsubscribable.Unsubscribe()
	})
}

// Unsubscribe disposes the resources held by the subscription. May, for
// instance, cancel an ongoing `Observable` execution or cancel any other
// type of work that started when the `Subscription` was created.
//
// This method is thread-safe. Finalizers are executed in sequence.
//
// Implements Unsuscribable.
func (s *subscriptionImpl) Unsubscribe() {
	s.mu.Lock()

	if s.done {
		s.mu.Unlock()
		return
	}

	s.done = true
	finals := s.finalizers
	ctxFinals := s.ctxFinalizers
	s.finalizers = nil
	s.ctxFinalizers = nil
	s.mu.Unlock()

	var errs []error

	// Execute simple teardowns
	for _, f := range finals {
		if err := execFinalizer(f); err != nil {
			errs = append(errs, err)
		}
	}

	// Execute context teardowns with a background context
	for _, f := range ctxFinals {
		if err := execFinalizerWithContext(f, context.Background()); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		panic(xerrors.Join(errs...))
	}
}

// UnsubscribeWithContext cancels the subscription and executes all registered
// teardown functions with the provided context. This allows cancellation-aware
// cleanup logic (e.g. context timeout or cancellation).

func (s *subscriptionImpl) UnsubscribeWithContext(ctx context.Context) {

	s.mu.Lock()

	if s.done {
		s.mu.Unlock()
		return
	}

	s.done = true
	finals := s.finalizers
	ctxFinals := s.ctxFinalizers
	s.finalizers = nil
	s.ctxFinalizers = nil
	s.mu.Unlock()

	var errs []error
	// Execute simple teardowns
	for _, f := range finals {
		if err := execFinalizer(f); err != nil {
			errs = append(errs, err)
		}
	}

	// Execute context teardowns with provided context
	for _, f := range ctxFinals {
		if err := execFinalizerWithContext(f, ctx); err != nil {
			errs = append(errs, err)
		}
	}

	if len(errs) > 0 {
		panic(xerrors.Join(errs...))
	}
}


// IsClosed returns true if the subscription has been disposed
// or if unsubscription is in progress.
//
// Implements Subscription.
func (s *subscriptionImpl) IsClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.done
}

// Wait blocks until a `Subscription` is canceled. It can be used for
// blocking until an `Observable` throws an error or completes.
//
// Please use it carefully. Calling this method is against the Reactive
// Programming Manifesto. This method might be deleted in the future.
//
// Note: using .Wait() is not recommended.
//
// Implements Subscription.
func (s *subscriptionImpl) Wait() {
	ch := make(chan struct{}, 1)

	// There is no guarantee that this callback will be the last finalizer
	// added to this subscription.
	s.Add(func() {
		ch <- struct{}{}
	})

	<-ch
	close(ch)
}

// execFinalizer runs the finalizer and catches any panics, converting them to errors.
func execFinalizer(finalizer func()) (err error) {
	lo.TryCatchWithErrorValue(
		func() error {
			finalizer()
			return nil
		},
		func(e any) {
			err = newUnsubscriptionError(recoverValueToError(e))
		},
	)

	return err
}

func execFinalizerWithContext(finalizer any, ctx context.Context) (err error) {
	switch f := finalizer.(type) {
	case func():
		return execFinalizer(f)
	case func(context.Context):
		lo.TryCatchWithErrorValue(
			func() error {
				f(ctx)
				return nil
			},
			func(e any) {
				err = newUnsubscriptionError(recoverValueToError(e))
			},
		)
	case TeardownWithContext:
		lo.TryCatchWithErrorValue(
			func() error {
				f(ctx)
				return nil
			},
			func(e any) {
				err = newUnsubscriptionError(recoverValueToError(e))
			},
		)
	}
	return err
}


// @TODO: Add methods Remove + RemoveSubscription.
// Currently, Go does not support function address comparison, so we cannot
// remove a finalizer from the list.
