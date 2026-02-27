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

package rotesting

import (
	"context"
	"testing"

	"github.com/samber/lo"
	"github.com/samber/ro"
)

// @TODO: Add new methods:
// - ExpectDurationEpsilon
// - ExpectDurationLessThan
// - ExpectDurationGreaterThan
// - ExpectDurationInRange

var _ AssertSpec[int] = (*assertImpl[int])(nil)

type assertImpl[T any] struct {
	t          *testing.T
	assertions []gotestingAssertion[T]
	source     ro.Observable[T]
}

type gotestingAssertion[T any] struct {
	notification ro.Notification[T]
	msgAndArgs   []any
}

// Assert creates a new instance of test. It is used to assert the behavior of an
// observable sequence.
//
// Inspired by Flux.
func Assert[T any](t *testing.T) AssertSpec[T] { //nolint:thelper
	return &assertImpl[T]{
		t:          t,
		assertions: []gotestingAssertion[T]{},
		source:     nil,
	}
}

func (t *assertImpl[T]) popAssertion() (gotestingAssertion[T], bool) {
	if len(t.assertions) == 0 {
		return gotestingAssertion[T]{}, false
	}

	assertion := t.assertions[0]
	t.assertions = t.assertions[1:]

	return assertion, true
}

func (t *assertImpl[T]) equal(expected, actual any, msgAndArgs ...any) bool {
	if expected == actual {
		return true
	}

	if len(msgAndArgs) > 0 {
		t.t.Errorf(msgAndArgs[0].(string), msgAndArgs[1:]...) //nolint:errcheck,forcetypeassert
	} else {
		t.t.Fail()
	}

	return false
}

func (t *assertImpl[T]) hasErrorOrCompletionNotification() bool {
	_, ok := lo.Find(t.assertions, func(assertion gotestingAssertion[T]) bool {
		return assertion.notification.Kind == ro.KindError || assertion.notification.Kind == ro.KindComplete
	})

	return ok
}

// Source sets the source observable to test.
func (t *assertImpl[T]) Source(source ro.Observable[T]) AssertSpec[T] {
	t.source = source
	return t
}

// ExpectNext expects the next value to be emitted by the source observable.
// It fails the test if the next value is not emitted. If the source observable
// emits an error or completes, it fails the test.
func (t *assertImpl[T]) ExpectNext(value T, msgAndArgs ...any) AssertSpec[T] {
	t.t.Helper()

	assertion := gotestingAssertion[T]{
		notification: ro.NewNotificationNext(value),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)

	return t
}

// ExpectNextSeq expects the next values to be emitted by the source observable.
// It fails the test if the next values are not emitted. If the source observable
// emits an error or completes, it fails the test.
func (t *assertImpl[T]) ExpectNextSeq(values ...T) AssertSpec[T] {
	t.t.Helper()

	for i := range values {
		assertion := gotestingAssertion[T]{
			notification: ro.NewNotificationNext(values[i]),
			// msgAndArgs:   []any{"expected '%v' value", (any)(values[i])},
		}
		t.assertions = append(t.assertions, assertion)
	}

	return t
}

// ExpectError expects the source observable to emit an error. It fails the test
// if the source observable emits a value or completes. If the source observable
// emits an error, it compares the error with the expected error. If the error
// is not equal to the expected error, it fails the test.
func (t *assertImpl[T]) ExpectError(err error, msgAndArgs ...any) AssertSpec[T] {
	t.t.Helper()

	if t.hasErrorOrCompletionNotification() {
		t.t.Fatal("cannot have multiple error or completion notifications")
	}

	assertion := gotestingAssertion[T]{
		notification: ro.NewNotificationError[T](err),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)

	return t
}

// ExpectComplete expects the source observable to complete. It fails the test
// if the source observable emits a value or an error.
func (t *assertImpl[T]) ExpectComplete(msgAndArgs ...any) AssertSpec[T] {
	t.t.Helper()

	if t.hasErrorOrCompletionNotification() {
		t.t.Fatal("cannot have multiple error or completion notifications")
	}

	assertion := gotestingAssertion[T]{
		notification: ro.NewNotificationComplete[T](),
		msgAndArgs:   msgAndArgs,
	}
	t.assertions = append(t.assertions, assertion)

	return t
}

// Verify subscribes to the source observable and verifies the assertions.
// It fails the test if the source observable emits a value, an error, or completes
// before all assertions are verified.
func (t *assertImpl[T]) Verify() {
	t.t.Helper()

	t.VerifyWithContext(context.Background())
}

// VerifyWithContext subscribes to the source observable and verifies the assertions.
// It fails the test if the source observable emits a value, an error, or completes
// before all assertions are verified.
func (t *assertImpl[T]) VerifyWithContext(ctx context.Context) {
	t.t.Helper()

	t.source.SubscribeWithContext(
		ctx,
		ro.NewObserverWithContext(
			func(ctx context.Context, value T) {
				assertion, ok := t.popAssertion()

				ok = ok && t.equal(ro.KindNext, assertion.notification.Kind, "expected '%s' notification, got 'Next'", assertion.notification.Kind)
				ok = ok && t.equal(assertion.notification.Value, value, assertion.msgAndArgs...)
				_ = ok
			},
			func(ctx context.Context, err error) {
				assertion, ok := t.popAssertion()

				ok = ok && t.equal(ro.KindError, assertion.notification.Kind, "expected '%s' notification, got 'Error'", assertion.notification.Kind)
				ok = ok && t.equal(assertion.notification.Err, err, assertion.msgAndArgs...)
				_ = ok
			},
			func(ctx context.Context) {
				assertion, ok := t.popAssertion()

				ok = ok && t.equal(ro.KindComplete, assertion.notification.Kind, "expected '%s' notification, got 'Complete'", assertion.notification.Kind)
				_ = ok
			},
		),
	)
}
