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

package rosort

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/constraints"
	"github.com/stretchr/testify/assert"
)

// Imported from "cmp" package. Introduced in go 1.21.
func isNaN[T constraints.Ordered](x T) bool {
	return x != x
}
func Compare[T constraints.Ordered](x, y T) int {
	xNaN := isNaN(x)
	yNaN := isNaN(y)
	if xNaN {
		if yNaN {
			return 0
		}
		return -1
	}
	if yNaN {
		return +1
	}
	if x < y {
		return -1
	}
	if x > y {
		return +1
	}
	return 0
}

func TestSort(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with empty observable
	values, err := ro.Collect(
		Sort(Compare[int])(
			ro.Just[int](),
		),
	)
	is.Equal([]int{}, values)
	is.Nil(err)

	// Test with single value
	values, err = ro.Collect(
		Sort(Compare[int])(
			ro.Just(42),
		),
	)
	is.Equal([]int{42}, values)
	is.Nil(err)

	// Test with already sorted values
	values, err = ro.Collect(
		Sort(Compare[int])(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with reverse sorted values
	values, err = ro.Collect(
		Sort(Compare[int])(
			ro.Just(5, 4, 3, 2, 1),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with mixed values
	values, err = ro.Collect(
		Sort(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)

	// Test with negative values
	values, err = ro.Collect(
		Sort(Compare[int])(
			ro.Just(-5, 10, -3, 0, 7, -1),
		),
	)
	is.Equal([]int{-5, -3, -1, 0, 7, 10}, values)
	is.Nil(err)

	// Test with strings
	valuesStr, err := ro.Collect(
		Sort(Compare[string])(
			ro.Just("banana", "apple", "cherry", "date"),
		),
	)
	is.Equal([]string{"apple", "banana", "cherry", "date"}, valuesStr)
	is.Nil(err)

	// Test with floats
	valuesFloat, err := ro.Collect(
		Sort(Compare[float64])(
			ro.Just(3.14, 2.71, 1.41, 2.23),
		),
	)
	is.Equal([]float64{1.41, 2.23, 2.71, 3.14}, valuesFloat)
	is.Nil(err)
}

func TestSortFunc(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with empty observable
	values, err := ro.Collect(
		SortFunc(Compare[int])(
			ro.Just[int](),
		),
	)
	is.Equal([]int{}, values)
	is.Nil(err)

	// Test with single value
	values, err = ro.Collect(
		SortFunc(Compare[int])(
			ro.Just(42),
		),
	)
	is.Equal([]int{42}, values)
	is.Nil(err)

	// Test with already sorted values
	values, err = ro.Collect(
		SortFunc(Compare[int])(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with reverse sorted values
	values, err = ro.Collect(
		SortFunc(Compare[int])(
			ro.Just(5, 4, 3, 2, 1),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with mixed values
	values, err = ro.Collect(
		SortFunc(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)

	// Test with custom comparison function (reverse order)
	values, err = ro.Collect(
		SortFunc(func(a, b int) int {
			return Compare(b, a) // reverse order
		})(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{5, 4, 3, 2, 1}, values)
	is.Nil(err)

	// Test with custom comparison function (absolute value)
	values, err = ro.Collect(
		SortFunc(func(a, b int) int {
			return Compare(abs(a), abs(b))
		})(
			ro.Just(-5, 3, -1, 4, -2),
		),
	)
	is.Equal([]int{-1, -2, 3, 4, -5}, values)
	is.Nil(err)

	// Test with strings using custom comparison (case insensitive)
	valuesStr, err := ro.Collect(
		SortFunc(func(a, b string) int {
			return Compare(toLower(a), toLower(b))
		})(
			ro.Just("Banana", "apple", "Cherry", "DATE"),
		),
	)
	is.Equal([]string{"apple", "Banana", "Cherry", "DATE"}, valuesStr)
	is.Nil(err)
}

func TestSortStableFunc(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with empty observable
	values, err := ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Just[int](),
		),
	)
	is.Equal([]int{}, values)
	is.Nil(err)

	// Test with single value
	values, err = ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Just(42),
		),
	)
	is.Equal([]int{42}, values)
	is.Nil(err)

	// Test with already sorted values
	values, err = ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with reverse sorted values
	values, err = ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Just(5, 4, 3, 2, 1),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Nil(err)

	// Test with mixed values
	values, err = ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)

	// Test with custom comparison function (reverse order)
	values, err = ro.Collect(
		SortStableFunc(func(a, b int) int {
			return Compare(b, a) // reverse order
		})(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{5, 4, 3, 2, 1}, values)
	is.Nil(err)

	// Test with custom comparison function (absolute value)
	values, err = ro.Collect(
		SortStableFunc(func(a, b int) int {
			return Compare(abs(a), abs(b))
		})(
			ro.Just(-5, 3, -1, 4, -2),
		),
	)
	is.Equal([]int{-1, -2, 3, 4, -5}, values)
	is.Nil(err)

	// Test with strings using custom comparison (case insensitive)
	valuesStr, err := ro.Collect(
		SortStableFunc(func(a, b string) int {
			return Compare(toLower(a), toLower(b))
		})(
			ro.Just("Banana", "apple", "Cherry", "DATE"),
		),
	)
	is.Equal([]string{"apple", "Banana", "Cherry", "DATE"}, valuesStr)
	is.Nil(err)
}

func TestSortWithError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with error observable
	values, err := ro.Collect(
		Sort(Compare[int])(
			ro.Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestSortFuncWithError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with error observable
	values, err := ro.Collect(
		SortFunc(Compare[int])(
			ro.Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestSortStableFuncWithError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with error observable
	values, err := ro.Collect(
		SortStableFunc(Compare[int])(
			ro.Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestSortWithContext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Test with context
	values, resultCtx, err := ro.CollectWithContext(ctx,
		Sort(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)
	is.NotNil(resultCtx)
}

func TestSortFuncWithContext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Test with context
	values, resultCtx, err := ro.CollectWithContext(ctx,
		SortFunc(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)
	is.NotNil(resultCtx)
}

func TestSortStableFuncWithContext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Test with context
	values, resultCtx, err := ro.CollectWithContext(ctx,
		SortStableFunc(Compare[int])(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{1, 1, 2, 3, 4, 5, 6, 9}, values)
	is.Nil(err)
	is.NotNil(resultCtx)
}

// Helper functions
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func toLower(s string) string {
	// Simple implementation for testing
	// In a real implementation, you'd use strings.ToLower
	result := ""
	for _, r := range s {
		if r >= 'A' && r <= 'Z' {
			result += string(r + 32)
		} else {
			result += string(r)
		}
	}
	return result
}

func TestReverse(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with empty observable
	values, err := ro.Collect(
		Reverse[int]()(
			ro.Just[int](),
		),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	// Test with single value
	values, err = ro.Collect(
		Reverse[int]()(
			ro.Just(42),
		),
	)
	is.Equal([]int{42}, values)
	is.NoError(err)

	// Test with multiple values
	values, err = ro.Collect(
		Reverse[int]()(
			ro.Just(1, 2, 3, 4, 5),
		),
	)
	is.Equal([]int{5, 4, 3, 2, 1}, values)
	is.NoError(err)

	// Test with strings
	valuesStr, err := ro.Collect(
		Reverse[string]()(
			ro.Just("apple", "banana", "cherry"),
		),
	)
	is.Equal([]string{"cherry", "banana", "apple"}, valuesStr)
	is.NoError(err)
}

func TestReverseWithError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Test with error observable
	values, err := ro.Collect(
		Reverse[int]()(
			ro.Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	// An error raised after some items were already buffered must not emit
	// any of the buffered items.
	values, err = ro.Collect(
		Reverse[int]()(
			ro.Pipe1(
				ro.Just(1, 2, 3),
				ro.ConcatWith(ro.Throw[int](assert.AnError)),
			),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestReverseWithContext(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	type ctxKey string
	key := ctxKey("index")

	var seen []int

	values, err := ro.Collect(
		ro.Pipe3(
			ro.Just(10, 20, 30),
			ro.MapIWithContext(func(ctx context.Context, v int, i int64) (context.Context, int) {
				return context.WithValue(ctx, key, int(i)), v
			}),
			Reverse[int](),
			ro.TapOnNextWithContext(func(ctx context.Context, _ int) {
				index, _ := ctx.Value(key).(int)
				seen = append(seen, index)
			}),
		),
	)
	is.Equal([]int{30, 20, 10}, values)
	is.Equal([]int{2, 1, 0}, seen)
	is.NoError(err)
}

func TestReverseWithContextCancellation(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	values, resultCtx, err := ro.CollectWithContext(ctx,
		Reverse[int]()(
			ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		),
	)
	is.Equal([]int{6, 2, 9, 5, 1, 4, 1, 3}, values)
	is.NoError(err)
	is.NotNil(resultCtx)
}

func TestReverseEarlyUnsubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var finalized int32

	obs := ro.Pipe2(
		ro.Never(),
		ro.TapOnFinalize[struct{}](func() {
			atomic.AddInt32(&finalized, 1)
		}),
		Reverse[struct{}](),
	)

	sub := obs.Subscribe(ro.OnNext(func(struct{}) {}))
	sub.Unsubscribe()

	is.EqualValues(1, atomic.LoadInt32(&finalized))
}

func testWithTimeout(t *testing.T, timeout time.Duration) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	done := make(chan struct{})
	go func() {
		defer close(done)
		<-ctx.Done()
	}()

	select {
	case <-done:
		t.Fatal("test timeout")
	default:
		// Continue with test
	}
}
