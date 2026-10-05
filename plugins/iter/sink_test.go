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

package roiter

import (
	"context"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestToSeq(t *testing.T) {
	// Create an observable that emits values
	observable := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
		for i := 1; i <= 5; i++ {
			observer.NextWithContext(ctx, i)
		}
		observer.CompleteWithContext(ctx)
		return nil
	})

	// Transform the observable into an iterator
	seq := ToSeq(observable)

	// Collect values from the iterator
	var values []int
	for v := range seq {
		values = append(values, v)
	}

	// Verify that all values were collected
	assert.Equal(t, []int{1, 2, 3, 4, 5}, values)

	t.Run("early break", func(t *testing.T) {
		// Create an observable that emits values
		observable := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
			for i := 1; i <= 10; i++ {
				select {
				case <-ctx.Done():
					return nil
				default:
					observer.NextWithContext(ctx, i)
				}
			}
			observer.CompleteWithContext(ctx)
			return nil
		})

		// Transform the observable into an iterator
		seq := ToSeq(observable)

		// Collect only the first 3 values
		var values []int
		count := 0
		for v := range seq {
			values = append(values, v)
			count++
			if count >= 3 {
				break
			}
		}

		// Verify that only 3 values were collected
		assert.Equal(t, []int{1, 2, 3}, values)
		assert.Equal(t, 3, len(values))
	})
}

func TestToSeq_withError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create an observable that emits an error
	observable := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
		observer.NextWithContext(ctx, 1)
		observer.NextWithContext(ctx, 2)
		observer.ErrorWithContext(ctx, assert.AnError)
		return nil
	})

	// Transform the observable into an iterator
	seq := ToSeq(observable)

	// Collect values from the iterator
	var values []int
	is.Panics(func() {
		for v := range seq {
			values = append(values, v)
		}
	})

	// Verify that values were collected before the error
	assert.Equal(t, []int{1, 2}, values)
}

func TestToSeq2(t *testing.T) {
	// Create an observable that emits key-value pairs
	observable := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[string]) ro.Teardown {
		observer.NextWithContext(ctx, "a")
		observer.NextWithContext(ctx, "b")
		observer.NextWithContext(ctx, "c")
		observer.CompleteWithContext(ctx)
		return nil
	})

	// Transform the observable into an iterator
	seq := ToSeq2(observable)

	// Collect key-value pairs from the iterator
	var keys []int
	var values []string
	for k, v := range seq {
		keys = append(keys, k)
		values = append(values, v)
	}

	// Verify that all key-value pairs were collected
	assert.Equal(t, []int{1, 2, 3}, keys)
	assert.Equal(t, []string{"a", "b", "c"}, values)
}

func TestToSeq2_withError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create an observable that emits key-value pairs and then an error
	observable := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[string]) ro.Teardown {
		observer.NextWithContext(ctx, "a")
		observer.NextWithContext(ctx, "b")
		observer.ErrorWithContext(ctx, assert.AnError)
		return nil
	})

	// Transform the observable into an iterator
	seq := ToSeq2(observable)

	// Collect key-value pairs from the iterator
	var keys []int
	var values []string

	is.Panics(func() {
		for k, v := range seq {
			keys = append(keys, k)
			values = append(values, v)
		}
	})

	// Verify that key-value pairs were collected before the error
	assert.Equal(t, []int{1, 2}, keys)
	assert.Equal(t, []string{"a", "b"}, values)
}
