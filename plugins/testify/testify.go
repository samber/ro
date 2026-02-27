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


package rotestify

import (
	"context"

	"github.com/samber/lo"
	"github.com/samber/ro"
	rotesting "github.com/samber/ro/testing"
	"github.com/stretchr/testify/assert"
)

type testify[T any] struct {
	is         *assert.Assertions
	assertions []testifyAssertion[T]
	source     ro.Observable[T]
}

type testifyAssertion[T any] struct {
	notification ro.Notification[T]
	msgAndArgs   []any
}

// Testify creates a new instance of test. It is used to assert the behavior of an
// observable sequence.
//
// Inspired by Flux.
func Testify[T any](is *assert.Assertions) rotesting.AssertSpec[T] {
	return &testify[T]{
		is:         is,
		assertions: []testifyAssertion[T]{},
		source:     nil,
	}
}

func (t *testify[T]) popAssertion() (testifyAssertion[T], bool) {
	if len(t.assertions) == 0 {
		return testifyAssertion[T]{}, false
	}

	assertion := t.assertions[0]
	t.assertions = t.assertions[1:]

	return assertion, true
}

func (t *testify[T]) hasErrorOrCompletionNotification() bool {
	_, ok := lo.Find(t.assertions, func(assertion testifyAssertion[T]) bool {
		return assertion.notification.Kind == ro.KindError || assertion.notification.Kind == ro.KindComplete
	})
	return ok
}

// Source sets the source observable for the test.
func (t *testify[T]) Source(source ro.Observable[T]) rotesting.AssertSpec[T] {
	t.source = source
	return t
}

// ExpectNext expects the next value to be emitted by the source observable.
// It fails the test if the next value is not emitted. If the source observable
// emits an error or completes, it fails the test.
func (t *testify[T]) ExpectNext(value T, msgAndArgs ...any) rotesting.AssertSpec[T] {
	assertion := testifyAssertion[T]{
		notification: ro.NewNotificationNext(value),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)
	return t
}

// ExpectNextSeq expects the next values to be emitted by the source observable.
// It fails the test if the next values are not emitted. If the source observable
// emits an error or completes, it fails the test.
func (t *testify[T]) ExpectNextSeq(values ...T) rotesting.AssertSpec[T] {
	for i := range values {
		assertion := testifyAssertion[T]{
			notification: ro.NewNotificationNext(values[i]),
			msgAndArgs:   []any{"expected '%v' value", (any)(values[i])},
		}
		t.assertions = append(t.assertions, assertion)
	}
	return t
}

// ExpectError expects the source observable to emit an error. It fails the test
// if the source observable emits a value or completes. If the source observable
// emits an error, it compares the error with the expected error. If the error
// is not equal to the expected error, it fails the test.
func (t *testify[T]) ExpectError(err error, msgAndArgs ...any) rotesting.AssertSpec[T] {
	if t.hasErrorOrCompletionNotification() {
		t.is.Fail("cannot have multiple error or completion notifications")
	}

	assertion := testifyAssertion[T]{
		notification: ro.NewNotificationError[T](err),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)
	return t
}

// ExpectComplete expects the source observable to complete. It fails the test
// if the source observable emits a value or an error.
func (t *testify[T]) ExpectComplete(msgAndArgs ...any) rotesting.AssertSpec[T] {
	if t.hasErrorOrCompletionNotification() {
		t.is.Fail("cannot have multiple error or completion notifications")
	}

	assertion := testifyAssertion[T]{
		notification: ro.NewNotificationComplete[T](),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)
	return t
}

// Verify subscribes to the source observable and verifies the assertions.
// It fails the test if the source observable emits a value, an error, or completes
// before all assertions are verified.
func (t *testify[T]) Verify() {
	t.VerifyWithContext(context.Background())
}

// Verify subscribes to the source observable and verifies the assertions.
// It fails the test if the source observable emits a value, an error, or completes
// before all assertions are verified.
func (t *testify[T]) VerifyWithContext(ctx context.Context) {
	t.source.SubscribeWithContext(
		ctx,
		ro.NewObserverWithContext(
			func(ctx context.Context, value T) {
				assertion, ok := t.popAssertion()

				ok = ok && t.is.Equal(ro.KindNext, assertion.notification.Kind, "expected '%s' notification, got 'Next'", assertion.notification.Kind)
				ok = ok && t.is.Equal(assertion.notification.Value, value, assertion.msgAndArgs...)
				_ = ok
			},
			func(ctx context.Context, err error) {
				assertion, ok := t.popAssertion()

				ok = ok && t.is.Equal(ro.KindError, assertion.notification.Kind, "expected '%s' notification, got 'Error'", assertion.notification.Kind)
				ok = ok && t.is.Equal(assertion.notification.Err, err, assertion.msgAndArgs...)
				_ = ok
			},
			func(ctx context.Context) {
				assertion, ok := t.popAssertion()

				ok = ok && t.is.Equal(ro.KindComplete, assertion.notification.Kind, "expected '%s' notification, got 'Complete'", assertion.notification.Kind)
				_ = ok
			},
		),
	)
}
