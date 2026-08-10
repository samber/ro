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
	"sort"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/constraints"
)

////////////////////////////////////////////////////////////
//
// This plugin is a wrapper around the sort package.
//
// The following operators has been added to a plugin
// instead of package, because we don't recommend to
// use it.
//
// The operators load into memory all the values of the
// observable before sorting them. This should not be used
// for large datasets.
//
////////////////////////////////////////////////////////////

// bufferedItem keeps the context an item was emitted with, so replaying a
// buffered item later preserves the exact same context chain, item per item.
type bufferedItem[T any] struct {
	ctx   context.Context
	value T
}

// sortBuffer buffers every item emitted by the source Observable and, on
// completion, re-emits them sorted with cmp. Nothing is emitted before the
// source completes. Each buffered item is replayed with the context it was
// originally emitted with.
func sortBuffer[T any](stable bool, cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			buffer := []bufferedItem[T]{}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buffer = append(buffer, bufferedItem[T]{ctx, value})
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						less := func(i, j int) bool {
							return cmp(buffer[i].value, buffer[j].value) < 0
						}

						if stable {
							sort.SliceStable(buffer, less)
						} else {
							sort.Slice(buffer, less)
						}

						for _, item := range buffer {
							destination.NextWithContext(item.ctx, item.value)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Sort sorts the observable values using the provided comparison function.
//
// Warning: the whole stream is held in memory. Never use it on an unbounded source.
// Play: https://go.dev/play/p/3hL6m9jK5nV
func Sort[T constraints.Ordered](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return sortBuffer(false, cmp)
}

// SortFunc sorts the observable values using the provided comparison function.
//
// Warning: the whole stream is held in memory. Never use it on an unbounded source.
// Play: https://go.dev/play/p/PzNTA9Vufy7
func SortFunc[T comparable](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return sortBuffer(false, cmp)
}

// SortStableFunc sorts the observable values using the provided comparison function.
// Unlike SortFunc, the relative order of equivalent elements (cmp(a, b) == 0) is
// preserved.
//
// Warning: the whole stream is held in memory. Never use it on an unbounded source.
// Play: https://go.dev/play/p/6b1tIxX9gfO
func SortStableFunc[T comparable](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return sortBuffer(true, cmp)
}

// Reverse buffers every item emitted by the source Observable and, on completion,
// re-emits them one by one in reverse order. Nothing is emitted before the source
// completes.
//
// Warning: the whole stream is held in memory. Never use it on an unbounded source.
func Reverse[T any]() func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			buffer := []bufferedItem[T]{}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buffer = append(buffer, bufferedItem[T]{ctx, value})
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						for i := len(buffer) - 1; i >= 0; i-- {
							destination.NextWithContext(buffer[i].ctx, buffer[i].value)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}
