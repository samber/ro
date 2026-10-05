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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOperatorContextContextWithValue(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type ctxKey string

	ctxKeyStart := ctxKey("start")
	ctxKey42 := ctxKey("42")
	ctxKey42Int := 42

	count := 0

	obs := Pipe4(
		Just(1, 2, 3, 4),
		ContextWithValue[int](ctxKey42Int, -42),
		ContextWithValue[int](ctxKey42, "42"),
		ContextWithValue[int](ctxKey42Int, 42),

		// generate an error after 4 messages
		MapErrWithContext(func(ctx context.Context, item int) (int, context.Context, error) {
			is.Equal("trats", ctx.Value(ctxKeyStart))
			is.Equal("42", ctx.Value(ctxKey42))
			is.Equal(42, ctx.Value(ctxKey42Int))

			if item == 4 {
				return item, ctx, assert.AnError
			}

			return item, ctx, nil
		}),
	)

	sub := obs.SubscribeWithContext(
		context.WithValue(context.Background(), ctxKeyStart, "trats"),
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				is.Equal("trats", ctx.Value(ctxKeyStart))
				is.Equal("42", ctx.Value(ctxKey42))
				is.Equal(42, ctx.Value(ctxKey42Int))

				count++
			},
			func(ctx context.Context, err error) {
				is.Equal("trats", ctx.Value(ctxKeyStart))
				is.Equal("42", ctx.Value(ctxKey42))
				is.Equal(42, ctx.Value(ctxKey42Int))
				is.Equal(assert.AnError, err)

				count += 10
			},
			func(ctx context.Context) {
				is.Fail("complete")
			},
		),
	)

	sub.Unsubscribe()

	is.Equal(13, count)

	t.Run("chained operators accumulate values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		type ctxKey string

		key1 := ctxKey("key1")
		key2 := ctxKey("key2")

		values := []int{}
		contexts := []context.Context{}

		obs := Pipe2(
			Just(1, 2, 3, 4, 5),
			ContextWithValue[int](key1, "value1"),
			ContextWithValue[int](key2, "value2"),
		)

		sub := obs.SubscribeWithContext(
			context.Background(),
			NewObserverWithContext(
				func(ctx context.Context, value int) {
					// Should have context values
					is.Equal("value1", ctx.Value(key1))
					is.Equal("value2", ctx.Value(key2))

					values = append(values, value)
					contexts = append(contexts, ctx)
				},
				func(ctx context.Context, err error) {
					is.Fail("should not error")
				},
				func(ctx context.Context) {
					// Should complete normally
				},
			),
		)

		sub.Unsubscribe()

		is.Equal([]int{1, 2, 3, 4, 5}, values)
		is.Len(contexts, 5)

		// Check that all contexts have the expected values
		for _, ctx := range contexts {
			is.Equal("value1", ctx.Value(key1))
			is.Equal("value2", ctx.Value(key2))
		}
	})
}

func TestOperatorContextContextWithTimeout(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	timeout := 100 * time.Millisecond
	values := []int{}

	obs := Pipe1(
		Just(1, 2, 3, 4, 5),
		ContextWithTimeout[int](timeout),
	)

	sub := obs.SubscribeWithContext(
		context.Background(),
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				// Check that the context has a deadline
				deadline, ok := ctx.Deadline()
				is.True(ok)
				is.True(deadline.After(time.Now()))
				is.True(deadline.Before(time.Now().Add(timeout + 10*time.Millisecond)))

				values = append(values, value)
			},
			func(ctx context.Context, err error) {
				is.Fail("should not error")
			},
			func(ctx context.Context) {
				// Should complete normally
			},
		),
	)

	sub.Unsubscribe()

	is.Equal([]int{1, 2, 3, 4, 5}, values)
}

// func TestOperatorContextContextWithTimeoutCause(t *testing.T) {
// 	t.Parallel()
// 	is := assert.New(t)

// 	timeout := 100 * time.Millisecond
// 	cause := assert.AnError
// 	values := []int{}

// 	obs := Pipe1(
// 		Just(1, 2, 3, 4, 5),
// 		ContextWithTimeoutCause[int](timeout, cause),
// 	)

// 	sub := obs.SubscribeWithContext(
// 		context.Background(),
// 		NewObserverWithContext(
// 			func(ctx context.Context, value int) {
// 				// Check that the context has a deadline
// 				deadline, ok := ctx.Deadline()
// 				is.True(ok)
// 				is.True(deadline.After(time.Now()))
// 				is.True(deadline.Before(time.Now().Add(timeout + 10*time.Millisecond)))

// 				values = append(values, value)
// 			},
// 			func(ctx context.Context, err error) {
// 				is.Fail("should not error")
// 			},
// 			func(ctx context.Context) {
// 				// Should complete normally
// 			},
// 		),
// 	)

// 	sub.Unsubscribe()

// 	is.Equal([]int{1, 2, 3, 4, 5}, values)
// }

func TestOperatorContextContextWithDeadline(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	deadline := time.Now().Add(100 * time.Millisecond)
	values := []int{}

	obs := Pipe1(
		Just(1, 2, 3, 4, 5),
		ContextWithDeadline[int](deadline),
	)

	sub := obs.SubscribeWithContext(
		context.Background(),
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				// Check that the context has the expected deadline
				ctxDeadline, ok := ctx.Deadline()
				is.True(ok)
				is.Equal(deadline, ctxDeadline)

				values = append(values, value)
			},
			func(ctx context.Context, err error) {
				is.Fail("should not error")
			},
			func(ctx context.Context) {
				// Should complete normally
			},
		),
	)

	sub.Unsubscribe()

	is.Equal([]int{1, 2, 3, 4, 5}, values)
}

// func TestOperatorContextContextWithDeadlineCause(t *testing.T) {
// 	t.Parallel()
// 	is := assert.New(t)

// 	deadline := time.Now().Add(100 * time.Millisecond)
// 	cause := assert.AnError
// 	values := []int{}

// 	obs := Pipe1(
// 		Just(1, 2, 3, 4, 5),
// 		ContextWithDeadlineCause[int](deadline, cause),
// 	)

// 	sub := obs.SubscribeWithContext(
// 		context.Background(),
// 		NewObserverWithContext(
// 			func(ctx context.Context, value int) {
// 				// Check that the context has the expected deadline
// 				ctxDeadline, ok := ctx.Deadline()
// 				is.True(ok)
// 				is.Equal(deadline, ctxDeadline)

// 				values = append(values, value)
// 			},
// 			func(ctx context.Context, err error) {
// 				is.Fail("should not error")
// 			},
// 			func(ctx context.Context) {
// 				// Should complete normally
// 			},
// 		),
// 	)

// 	sub.Unsubscribe()

// 	is.Equal([]int{1, 2, 3, 4, 5}, values)
// }

func TestOperatorContextContextReset(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type ctxKey string

	originalKey := ctxKey("original")
	newKey := ctxKey("new")

	originalCtx := context.WithValue(context.Background(), originalKey, "original_value")
	newCtx := context.WithValue(context.Background(), newKey, "new_value")

	values := []int{}
	contexts := []context.Context{}

	obs := Pipe1(
		Just(1, 2, 3, 4, 5),
		ContextReset[int](newCtx),
	)

	sub := obs.SubscribeWithContext(
		originalCtx,
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				// Should have new context values, not original
				is.Nil(ctx.Value(originalKey))
				is.Equal("new_value", ctx.Value(newKey))

				values = append(values, value)
				contexts = append(contexts, ctx)
			},
			func(ctx context.Context, err error) {
				is.Fail("should not error")
			},
			func(ctx context.Context) {
				// Should complete normally
			},
		),
	)

	sub.Unsubscribe()

	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Len(contexts, 5)

	for _, ctx := range contexts {
		is.Nil(ctx.Value(originalKey))
		is.Equal("new_value", ctx.Value(newKey))
	}

	t.Run("nil context falls back to background", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		type ctxKey string

		originalKey := ctxKey("original")

		originalCtx := context.WithValue(context.Background(), originalKey, "original_value")

		values := []int{}
		contexts := []context.Context{}

		obs := Pipe1(
			Just(1, 2, 3, 4, 5),
			ContextReset[int](nil), //nolint:staticcheck
		)

		sub := obs.SubscribeWithContext(
			originalCtx,
			NewObserverWithContext(
				func(ctx context.Context, value int) {
					// Should have background context, not original
					is.Nil(ctx.Value(originalKey))

					values = append(values, value)
					contexts = append(contexts, ctx)
				},
				func(ctx context.Context, err error) {
					is.Fail("should not error")
				},
				func(ctx context.Context) {
					// Should complete normally
				},
			),
		)

		sub.Unsubscribe()

		is.Equal([]int{1, 2, 3, 4, 5}, values)
		is.Len(contexts, 5)

		for _, ctx := range contexts {
			is.Nil(ctx.Value(originalKey))
		}
	})
}

func TestOperatorContextContextMap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type ctxKey string

	key1 := ctxKey("key1")

	values := []int{}
	contexts := []context.Context{}

	obs := Pipe1(
		Just(1, 2, 3, 4, 5),
		ContextMap[int](func(ctx context.Context) context.Context {
			// Add a new key-value pair to the context
			return context.WithValue(ctx, key1, "mapped_value")
		}),
	)

	sub := obs.SubscribeWithContext(
		context.WithValue(context.Background(), key1, "original_value"),
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				// Should have mapped context value
				is.Equal("mapped_value", ctx.Value(key1))

				values = append(values, value)
				contexts = append(contexts, ctx)
			},
			func(ctx context.Context, err error) {
				is.Fail("should not error")
			},
			func(ctx context.Context) {
				// Should complete normally
			},
		),
	)

	sub.Unsubscribe()

	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Len(contexts, 5)

	for _, ctx := range contexts {
		is.Equal("mapped_value", ctx.Value(key1))
	}
}

func TestOperatorContextContextMapI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type ctxKey string

	key1 := ctxKey("key1")
	indexKey := ctxKey("index")

	values := []int{}
	contexts := []context.Context{}

	obs := Pipe1(
		Just(1, 2, 3, 4, 5),
		ContextMapI[int](func(ctx context.Context, index int64) context.Context {
			// Add index to the context
			return context.WithValue(ctx, indexKey, index)
		}),
	)

	sub := obs.SubscribeWithContext(
		context.WithValue(context.Background(), key1, "original_value"),
		NewObserverWithContext(
			func(ctx context.Context, value int) {
				// Should have original context value and index
				is.Equal("original_value", ctx.Value(key1))
				index := ctx.Value(indexKey)
				is.NotNil(index)

				values = append(values, value)
				contexts = append(contexts, ctx)
			},
			func(ctx context.Context, err error) {
				is.Fail("should not error")
			},
			func(ctx context.Context) {
				// Should complete normally
			},
		),
	)

	sub.Unsubscribe()

	is.Equal([]int{1, 2, 3, 4, 5}, values)
	is.Len(contexts, 5)

	// Check that each context has the correct index
	for i, ctx := range contexts {
		is.Equal(int64(i), ctx.Value(indexKey))
	}
}

func TestOperatorContextThrowOnContextCancel(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values := []int{}

	obs := Pipe1(
		NewObservableWithContext(func(ctx context.Context, destination Observer[int64]) Teardown {
			time.Sleep(50 * time.Millisecond)
			destination.NextWithContext(ctx, 0)

			time.Sleep(50 * time.Millisecond)
			destination.NextWithContext(ctx, 1)

			time.Sleep(50 * time.Millisecond)
			c, cancel := context.WithCancel(ctx)
			cancel()
			destination.NextWithContext(c, 2)

			time.Sleep(50 * time.Millisecond)
			destination.NextWithContext(ctx, 3)

			destination.CompleteWithContext(ctx)
			return nil
		}),
		ThrowOnContextCancel[int64](),
	)

	sub := obs.Subscribe(
		NewObserverWithContext(
			func(ctx context.Context, value int64) {
				values = append(values, int(value))
			},
			func(ctx context.Context, err error) {
				is.Equal(context.Canceled, err)
				is.Equal([]int{0, 1}, values)
			},
			func(ctx context.Context) {
				// Should not complete normally if context is canceled
				is.Fail("should not complete normally")
			},
		),
	)
	defer sub.Unsubscribe()

	is.Equal([]int{0, 1}, values)
}

// func TestOperatorContextThrowOnContextCancelWithTimeout(t *testing.T) {
// 	t.Parallel()
// 	is := assert.New(t)

// 	values := []int{}

// 	obs := Pipe2(
// 		Interval(10*time.Millisecond),
// 		ContextWithTimeout[IntervalValue](25*time.Millisecond), // Very short timeout
// 		ThrowOnContextCancel[IntervalValue](),
// 	)

// 	sub := obs.SubscribeWithContext(
// 		context.Background(),
// 		NewObserverWithContext(
// 			func(ctx context.Context, value IntervalValue) {
// 				values = append(values, int(value.Value))
// 			},
// 			func(ctx context.Context, err error) {
// 				is.Equal(context.DeadlineExceeded, err)
// 				is.Equal([]int{0, 1, 2}, values)
// 			},
// 			func(ctx context.Context) {
// 				// Should not complete normally if context times out
// 				is.Fail("should not complete normally")
// 			},
// 		),
// 	)

// 	// Wait for the timeout to occur
// 	time.Sleep(100 * time.Millisecond)
// 	sub.Unsubscribe()

// 	// Should have received some values before timeout
// 	is.Equal([]int{0, 1, 2}, values)
// }
