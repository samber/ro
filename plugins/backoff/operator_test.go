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

package robackoff

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// helper: source that fails the first n times, then succeeds with values.
func failThenSucceed(failCount int, values ...int) ro.Observable[int] {
	calls := 0
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
		calls++
		if calls <= failCount {
			dst.ErrorWithContext(ctx, errors.New("transient error"))
			return nil
		}
		for _, v := range values {
			dst.NextWithContext(ctx, v)
		}
		dst.CompleteWithContext(ctx)
		return nil
	})
}

// --- Retry (convenience wrapper) ---

func TestRetryPluginOperatorEmpty(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(
		Retry[int](nil)(ro.Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorNoError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(
		Retry[int](nil)(ro.Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorPropagatesError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// No retries configured via Retry(), so it retries forever — cap with MaxRetries.
	values, err := ro.Collect(
		RetryWithConfig[int](Config{MaxRetries: 1})(ro.Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

// --- RetryWithConfig ---

func TestRetryPluginOperatorMaxRetriesExhausted(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sentinel := errors.New("always fails")
	values, err := ro.Collect(
		RetryWithConfig[int](Config{MaxRetries: 3})(ro.Throw[int](sentinel)),
	)
	is.Equal([]int{}, values)
	is.Equal(sentinel, err)
}

func TestRetryPluginOperatorSucceedsBeforeMaxRetries(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	src := failThenSucceed(2, 10, 20, 30)
	values, err := ro.Collect(
		RetryWithConfig[int](Config{MaxRetries: 5})(src),
	)
	is.Equal([]int{10, 20, 30}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorUnlimitedRetries(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// MaxRetries == 0 means unlimited; source succeeds after 10 failures.
	src := failThenSucceed(10, 42)
	values, err := ro.Collect(
		RetryWithConfig[int](Config{MaxRetries: 0})(src),
	)
	is.Equal([]int{42}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorWithDelay(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	src := failThenSucceed(1, 7)
	values, err := ro.Collect(
		RetryWithConfig[int](Config{
			MaxRetries: 3,
			Strategy:   Constant(1 * time.Millisecond),
		})(src),
	)
	is.Equal([]int{7}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorContextCancellation(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	values, _, err := ro.CollectWithContext(
		ctx,
		RetryWithConfig[int](Config{
			MaxRetries: 10,
			Strategy:   Constant(1 * time.Hour), // would block forever without cancel
		})(ro.Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.Error(err) // context.Canceled or the source error
}

func TestRetryPluginOperatorResetOnSuccess(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	src := failThenSucceed(1, 1, 2, 3)
	values, err := ro.Collect(
		RetryWithConfig[int](Config{
			MaxRetries:     2,
			ResetOnSuccess: true,
		})(src),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
}

func TestRetryPluginOperatorNoRetriesOnSuccess(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// When source completes without error, no retry should happen.
	callCount := 0
	src := ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
		callCount++
		dst.NextWithContext(ctx, 99)
		dst.CompleteWithContext(ctx)
		return nil
	})

	values, err := ro.Collect(
		RetryWithConfig[int](Config{MaxRetries: 5})(src),
	)
	is.Equal([]int{99}, values)
	is.NoError(err)
	is.Equal(1, callCount, "source should be subscribed exactly once on success")
}
