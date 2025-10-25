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

// Sort sorts the observable values using the provided comparison function.
// Play: https://go.dev/play/p/3hL6m9jK5nV
func Sort[T constraints.Ordered](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			values, ctx, err := ro.CollectWithContext(subscriberCtx, source)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			sort.Slice(values, func(i, j int) bool {
				return cmp(values[i], values[j]) < 0
			})

			for _, value := range values {
				destination.NextWithContext(ctx, value)
			}
			destination.CompleteWithContext(ctx)

			return nil
		})
	}
}

// SortFunc sorts the observable values using the provided comparison function.
// Play: https://go.dev/play/p/PzNTA9Vufy7
func SortFunc[T comparable](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			values, ctx, err := ro.CollectWithContext(subscriberCtx, source)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			sort.Slice(values, func(i, j int) bool {
				return cmp(values[i], values[j]) < 0
			})

			for _, value := range values {
				destination.NextWithContext(ctx, value)
			}
			destination.CompleteWithContext(ctx)

			return nil
		})
	}
}

// SortStableFunc sorts the observable values using the provided stable comparison function.
// Play: https://go.dev/play/p/6b1tIxX9gfO
func SortStableFunc[T comparable](cmp func(a, b T) int) func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			values, ctx, err := ro.CollectWithContext(subscriberCtx, source)
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return nil
			}

			sort.Slice(values, func(i, j int) bool {
				return cmp(values[i], values[j]) < 0
			})

			for _, value := range values {
				destination.NextWithContext(ctx, value)
			}
			destination.CompleteWithContext(ctx)

			return nil
		})
	}
}
