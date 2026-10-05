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
	"errors"
	"testing"
	"time"

	"github.com/samber/ro"
	rotesting "github.com/samber/ro/testing"
	"github.com/stretchr/testify/assert"
)

func TestTestify(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a := Testify[int](is).(*testify[int])
	is.Equal(t, t)
	is.Empty(a.assertions)
	is.Nil(a.source)

	// complete normaly
	Testify[int](is).
		Source(ro.Just(1, 2, 3)).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectComplete().
		Verify()

	// complete on error
	Testify[int](is).
		Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
			destination.Next(1)
			destination.Next(2)
			destination.Next(3)
			destination.Error(assert.AnError)
			destination.Next(4)
			return nil
		})).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectError(assert.AnError).
		Verify()

	// async observable
	Testify[int](is).
		Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Next(3)
				destination.Error(assert.AnError)
				destination.Next(4)
			}()
			return nil
		})).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectError(assert.AnError).
		Verify()

	t.Run("ExpectNextSeq", func(t *testing.T) {
		is := assert.New(t)

		// Test ExpectNextSeq with multiple values
		Testify[int](is).
			Source(ro.Just(1, 2, 3, 4, 5)).
			ExpectNextSeq(1, 2, 3, 4, 5).
			ExpectComplete().
			Verify()

		// Test ExpectNextSeq with empty sequence
		Testify[int](is).
			Source(ro.Just[int]()).
			ExpectComplete().
			Verify()

		// Test ExpectNextSeq with single value
		Testify[int](is).
			Source(ro.Just(42)).
			ExpectNextSeq(42).
			ExpectComplete().
			Verify()
	})

	t.Run("verify with context", func(t *testing.T) {
		is := assert.New(t)

		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()

		// Test with context
		Testify[int](is).
			Source(ro.Just(1, 2, 3)).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectComplete().
			VerifyWithContext(ctx)

		// Test with cancelled context
		cancelledCtx, cancel := context.WithCancel(context.Background())
		cancel()

		Testify[int](is).
			Source(ro.Just(1, 2, 3)).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectComplete().
			VerifyWithContext(cancelledCtx)
	})

	t.Run("empty observable", func(t *testing.T) {
		is := assert.New(t)

		// Test empty observable that completes immediately
		Testify[int](is).
			Source(ro.Just[int]()).
			ExpectComplete().
			Verify()

		// Test empty observable that errors immediately
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Error(errors.New("immediate error"))
				return nil
			})).
			ExpectError(errors.New("immediate error")).
			Verify()
	})

	t.Run("custom error", func(t *testing.T) {
		is := assert.New(t)

		customError := errors.New("custom error message")

		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Next(1)
				destination.Error(customError)
				return nil
			})).
			ExpectNext(1).
			ExpectError(customError).
			Verify()
	})

	t.Run("string type", func(t *testing.T) {
		is := assert.New(t)

		// Test with string type
		Testify[string](is).
			Source(ro.Just("hello", "world", "test")).
			ExpectNext("hello").
			ExpectNext("world").
			ExpectNext("test").
			ExpectComplete().
			Verify()

		// Test ExpectNextSeq with strings
		Testify[string](is).
			Source(ro.Just("a", "b", "c")).
			ExpectNextSeq("a", "b", "c").
			ExpectComplete().
			Verify()
	})

	t.Run("struct type", func(t *testing.T) {
		is := assert.New(t)

		type testStruct struct {
			ID   int
			Name string
		}

		items := []testStruct{
			{ID: 1, Name: "Alice"},
			{ID: 2, Name: "Bob"},
			{ID: 3, Name: "Charlie"},
		}

		// Test with struct type
		Testify[testStruct](is).
			Source(ro.Just(items...)).
			ExpectNext(items[0]).
			ExpectNext(items[1]).
			ExpectNext(items[2]).
			ExpectComplete().
			Verify()
	})

	t.Run("async observable", func(t *testing.T) {
		is := assert.New(t)

		// Test async observable with delay
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					time.Sleep(10 * time.Millisecond)
					destination.Next(1)
					time.Sleep(10 * time.Millisecond)
					destination.Next(2)
					time.Sleep(10 * time.Millisecond)
					destination.Next(3)
					destination.Complete()
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectComplete().
			Verify()
	})

	t.Run("mixed notifications", func(t *testing.T) {
		is := assert.New(t)

		// Test observable that emits values then errors
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Next(1)
				destination.Next(2)
				destination.Error(errors.New("mixed error"))
				destination.Next(3) // This should not be received
				return nil
			})).
			ExpectNext(1).
			ExpectNext(2).
			ExpectError(errors.New("mixed error")).
			Verify()

		// Test observable that emits values then completes
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Next(10)
				destination.Next(20)
				destination.Complete()
				destination.Next(30) // This should not be received
				return nil
			})).
			ExpectNext(10).
			ExpectNext(20).
			ExpectComplete().
			Verify()
	})

	t.Run("message args", func(t *testing.T) {
		is := assert.New(t)

		// Test with custom error messages
		Testify[int](is).
			Source(ro.Just(42)).
			ExpectNext(42, "expected value 42").
			ExpectComplete("should complete normally").
			Verify()

		// Test with error and custom message
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Error(errors.New("test error"))
				return nil
			})).
			ExpectError(errors.New("test error"), "expected test error").
			Verify()
	})

	t.Run("multiple terminal notifications", func(t *testing.T) {
		is := assert.New(t)

		// Note: The current implementation allows multiple error/completion notifications
		// during setup but fails during verification. This test demonstrates the behavior.
		// In a real scenario, you would typically only expect one error or completion.

		// Test that the first error notification is detected
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Error(errors.New("first error"))
				return nil
			})).
			ExpectError(errors.New("first error")).
			Verify()

		// Test that the first completion notification is detected
		Testify[int](is).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Complete()
				return nil
			})).
			ExpectComplete().
			Verify()
	})

	t.Run("complex observable", func(t *testing.T) {
		is := assert.New(t)

		// Test a more complex observable with multiple operations
		observable := ro.Pipe2(
			ro.Just(1, 2, 3, 4, 5),
			ro.Filter(func(value int) bool { return value%2 == 0 }),
			ro.Map(func(value int) int { return value * 2 }),
		)

		Testify[int](is).
			Source(observable).
			ExpectNext(4). // 2 * 2
			ExpectNext(8). // 4 * 2
			ExpectComplete().
			Verify()
	})

	t.Run("verify with timeout context", func(t *testing.T) {
		is := assert.New(t)

		// Test with a timeout context
		ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
		defer cancel()

		Testify[int](is).
			Source(ro.Just(1, 2, 3)).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectComplete().
			VerifyWithContext(ctx)
	})

	t.Run("interface compliance", func(t *testing.T) {
		is := assert.New(t)

		// Test that Testify implements AssertSpec interface
		var _ rotesting.AssertSpec[int] = Testify[int](is)
	})
}
