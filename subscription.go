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
	"sync"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/xerrors"
)

// Teardown is a function that cleans up resources, such as closing
// a file or a network connection. It is called when the Subscription is closed.
// It is part of a Subscription, and is returned by the Observable creation.
// It will be called only once, when the Subscription is canceled.
type Teardown func()

// Unsubscribable represents any type that can be unsubscribed from.
// It provides a common interface for cancellation operations.
type Unsubscribable interface {
	Unsubscribe()
}

// Subscription represents an ongoing execution of an `Observable`, and has
// a minimal API which allows you to cancel that execution.
type Subscription interface {
	Unsubscribable

	Add(teardown Teardown)
	AddUnsubscribable(unsubscribable Unsubscribable)
	IsClosed() bool
	Wait() // Note: using .Wait() is not recommended.
}

var _ Subscription = (*subscriptionImpl)(nil)

// NewSubscription creates a new Subscription. When `teardown` is nil, nothing
// is added. When the subscription is already disposed, the `teardown` callback
// is triggered immediately.
func NewSubscription(teardown Teardown) Subscription {
	teardowns := make([]func(), 0, 4) // Pre-allocate for common case
	if teardown != nil {
		teardowns = append(teardowns, teardown)
	}

	return &subscriptionImpl{
		done:       false,
		mu:         sync.Mutex{},
		finalizers: teardowns,
	}
}

type subscriptionImpl struct {
	done       bool
	mu         sync.Mutex // Should be a RWMutex because of the .IsClosed() method, but sync.RWMutex is 30% slower.
	finalizers []func()
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
		teardown() // not protected against panics
	} else {
		s.finalizers = append(s.finalizers, teardown)
	}
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

	s.Add(unsubscribable.Unsubscribe)
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

	if len(s.finalizers) == 0 {
		s.mu.Unlock()
		return
	}

	finalizers := s.finalizers
	s.finalizers = make([]func(), 0)
	s.mu.Unlock()

	var errs []error

	// Note: we prefer not running this in parallel.
	for i := range finalizers {
		err := execFinalizer(finalizers[i]) // protected against panics
		if err != nil {
			// OnUnhandledError(err)
			errs = append(errs, err)
		}
	}

	// Error is triggered after the recursive call to finalizers
	// because we want to execute all finalizers before panicking.
	if len(errs) > 0 {
		// errors.Join has been introduced in go 1.20
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

			err = nil

			return nil
		},
		func(e any) {
			err = newUnsubscriptionError(recoverValueToError(e))
		},
	)

	return err
}

// @TODO: Add methods Remove + RemoveSubscription.
// Currently, Go does not support function address comparison, so we cannot
// remove a finalizer from the list.
