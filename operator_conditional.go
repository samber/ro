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

	"github.com/samber/lo"
)

// All determines whether all elements of an observable sequence satisfy a condition.
// Play: https://go.dev/play/p/t22F_crlA-l
func All[T any](predicate func(T) bool) func(Observable[T]) Observable[bool] {
	return AllIWithContext(func(ctx context.Context, v T, _ int64) bool {
		return predicate(v)
	})
}

// AllWithContext determines whether all elements of an observable sequence satisfy a condition.
// Play: https://go.dev/play/p/NEA7Zi7yVNh
func AllWithContext[T any](predicate func(ctx context.Context, item T) bool) func(Observable[T]) Observable[bool] {
	return AllIWithContext(func(ctx context.Context, item T, _ int64) bool {
		return predicate(ctx, item)
	})
}

// AllI determines whether all elements of an observable sequence satisfy a condition.
func AllI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[bool] {
	return AllIWithContext(func(ctx context.Context, item T, index int64) bool {
		return predicate(item, index)
	})
}

// AllIWithContext determines whether all elements of an observable sequence satisfy a condition.
// Play: https://go.dev/play/p/UkOzE4wQXPG
func AllIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool) func(Observable[T]) Observable[bool] {
	return func(source Observable[T]) Observable[bool] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[bool]) Teardown {
			ok := true
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if ok {
							ok = predicate(ctx, value, i)
							i++
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, ok)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Contains determines whether an observable sequence contains a specified element with an equality comparer.
// Play: https://go.dev/play/p/ldteqqGsMWM
func Contains[T any](predicate func(item T) bool) func(Observable[T]) Observable[bool] {
	return ContainsI(func(v T, _ int64) bool {
		return predicate(v)
	})
}

// ContainsWithContext determines whether an observable sequence contains a specified element with an equality comparer.
// Play: https://go.dev/play/p/RPHkyiLrFVW
func ContainsWithContext[T any](predicate func(ctx context.Context, item T) bool) func(Observable[T]) Observable[bool] {
	return ContainsIWithContext(func(ctx context.Context, v T, _ int64) bool {
		return predicate(ctx, v)
	})
}

// ContainsI determines whether an observable sequence contains a specified element with an equality comparer.
func ContainsI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[bool] {
	return ContainsIWithContext(func(ctx context.Context, v T, i int64) bool {
		return predicate(v, i)
	})
}

// ContainsIWithContext determines whether an observable sequence contains a specified element with an equality comparer.
// Play: https://go.dev/play/p/TkLfujMVNJb
func ContainsIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool) func(Observable[T]) Observable[bool] {
	return func(source Observable[T]) Observable[bool] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[bool]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ok := predicate(ctx, value, i)
						if ok {
							destination.NextWithContext(ctx, ok)
							destination.CompleteWithContext(ctx)
						}

						i++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, false)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Find returns the first element of an observable sequence that satisfies the condition.
// Play: https://go.dev/play/p/2f5rn0HoKeq
func Find[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return FindI(func(item T, _ int64) bool {
		return predicate(item)
	})
}

// FindWithContext returns the first element of an observable sequence that satisfies the condition.
// Play: https://go.dev/play/p/BVm-Grgv11w
func FindWithContext[T any](predicate func(ctx context.Context, item T) bool) func(Observable[T]) Observable[T] {
	return FindIWithContext(func(ctx context.Context, v T, _ int64) bool {
		return predicate(ctx, v)
	})
}

// FindI returns the first element of an observable sequence that satisfies the condition.
func FindI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return FindIWithContext(func(ctx context.Context, v T, i int64) bool {
		return predicate(v, i)
	})
}

// FindIWithContext returns the first element of an observable sequence that satisfies the condition.
// Play: https://go.dev/play/p/X8oT_CF9IKM
func FindIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ok := predicate(ctx, value, i)
						if ok {
							destination.NextWithContext(ctx, value)
							destination.CompleteWithContext(ctx)
						}

						i++
					},
					destination.ErrorWithContext,
					// return zero value or error ?
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Iif determines which one of two observables to return based on a condition.
// Play: https://go.dev/play/p/t-sNgL5EZA-
func Iif[T any](predicate func() bool, source1, source2 Observable[T]) func() Observable[T] {
	return func() Observable[T] {
		if predicate() {
			return source1
		}

		return source2
	}
}

// DefaultIfEmpty emits a default value if the source observable emits no items.
// Play: https://go.dev/play/p/WDh807OLPWv
func DefaultIfEmpty[T any](defaultValue T) func(Observable[T]) Observable[T] {
	return DefaultIfEmptyWithContext(context.Background(), defaultValue)
}

// DefaultIfEmptyWithContext emits a default value if the source observable emits no items.
func DefaultIfEmptyWithContext[T any](defaultCtx context.Context, defaultValue T) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			empty := true

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						empty = false

						destination.NextWithContext(ctx, value)
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if empty {
							destination.NextWithContext(defaultCtx, defaultValue)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SequenceEqual determines whether two observable sequences are equal by comparing the elements pairwise.
// Play: https://go.dev/play/p/cBIQlH01byQ
func SequenceEqual[T comparable](obsB Observable[T]) func(Observable[T]) Observable[bool] {
	return func(source Observable[T]) Observable[bool] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[bool]) Teardown {
			sub := Zip2(source, obsB).
				SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, values lo.Tuple2[T, T]) {
							if values.A != values.B {
								destination.NextWithContext(ctx, false)
								destination.CompleteWithContext(ctx)
							}
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							destination.NextWithContext(ctx, true)
							destination.CompleteWithContext(ctx)
						},
					),
				)

			return sub.Unsubscribe
		})
	}
}
