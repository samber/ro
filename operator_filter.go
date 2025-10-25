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
	"sync/atomic"

	"github.com/samber/lo"
)

// Filter emits only those items from an Observable that pass a predicate test.
// Play: https://go.dev/play/p/3UsEzgLAp4s
func Filter[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return FilterIWithContext(func(ctx context.Context, v T, _ int64) (context.Context, bool) {
		return ctx, predicate(v)
	})
}

// FilterWithContext emits only those items from an Observable that pass a predicate test.
func FilterWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return FilterIWithContext(func(ctx context.Context, v T, index int64) (context.Context, bool) {
		return predicate(ctx, v)
	})
}

// FilterI emits only those items from an Observable that pass a predicate test.
func FilterI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return FilterIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return ctx, predicate(v, i)
	})
}

// FilterIWithContext emits only those items from an Observable that pass a predicate test.
func FilterIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ctx, ok := predicate(ctx, value, i)
						if ok {
							destination.NextWithContext(ctx, value)
						}

						i++
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Distinct suppresses duplicate items in an Observable.
func Distinct[T comparable]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			seen := map[T]struct{}{}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if _, ok := seen[value]; !ok {
							destination.NextWithContext(ctx, value)

							seen[value] = struct{}{}
						}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// DistinctBy suppresses duplicate items in an Observable based on a key selector.
func DistinctBy[T any, K comparable](keySelector func(item T) K) func(Observable[T]) Observable[T] {
	return DistinctByWithContext(func(ctx context.Context, item T) (context.Context, K) {
		return ctx, keySelector(item)
	})
}

// DistinctByWithContext suppresses duplicate items in an Observable based on a key selector.
// The context is passed to the key selector function.
func DistinctByWithContext[T any, K comparable](keySelector func(ctx context.Context, item T) (context.Context, K)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			seen := map[K]struct{}{}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ctx, key := keySelector(ctx, value)
						if _, ok := seen[key]; !ok {
							destination.NextWithContext(ctx, value)

							seen[key] = struct{}{}
						}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// IgnoreElements does not emit any items from an Observable but mirrors its
// termination notification. It is useful for ignoring all the items from an
// Observable but you want to be notified when it completes or when it throws an error.
func IgnoreElements[T any]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Skip suppresses the first n items emitted by an Observable.
// If the count is greater than the number of items emitted by the source Observable,
// Skip will not emit any items. If the count is zero, Skip will emit all items.
func Skip[T any](count int64) func(Observable[T]) Observable[T] {
	if count < 0 {
		panic(ErrSkipWrongCount)
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			index := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if index >= count {
							destination.NextWithContext(ctx, value)
						}

						index++
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SkipWhile skips items emitted by an Observable until a specified condition
// becomes false. It will then emit all the subsequent items. If the condition
// is never false, SkipWhile will not emit any items. If the condition is false
// on the first item, SkipWhile will emit all items.
func SkipWhile[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return SkipWhileI(func(v T, i int64) bool {
		return predicate(v)
	})
}

// SkipWhileWithContext skips items emitted by an Observable until a specified condition
// becomes false. It will then emit all the subsequent items. If the condition
// is never false, SkipWhile will not emit any items. If the condition is false
// on the first item, SkipWhile will emit all items.
func SkipWhileWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return SkipWhileIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return predicate(ctx, v)
	})
}

// SkipWhileI skips items emitted by an Observable until a specified condition
// becomes false. It will then emit all the subsequent items. If the condition
// is never false, SkipWhile will not emit any items. If the condition is false
// on the first item, SkipWhile will emit all items.
func SkipWhileI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return SkipWhileIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return ctx, predicate(v, i)
	})
}

// SkipWhileIWithContext skips items emitted by an Observable until a specified condition
// becomes false. It will then emit all the subsequent items. If the condition
// is never false, SkipWhile will not emit any items. If the condition is false
// on the first item, SkipWhile will emit all items.
func SkipWhileIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			skipping := true
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if !skipping {
							destination.NextWithContext(ctx, value)
						} else if newCtx, ok := predicate(ctx, value, i); !ok {
							skipping = false

							destination.NextWithContext(newCtx, value)
						}

						i++
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SkipLast suppresses the last n items emitted by an Observable. If the count
// is greater than the number of items emitted by the source Observable, SkipLast
// will not emit any items. If the count is zero, SkipLast will emit all items.
func SkipLast[T any](count int) func(Observable[T]) Observable[T] {
	if count < 1 {
		panic(ErrSkipLastWrongCount)
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			// Use a circular buffer approach to avoid memory allocations
			buffer := make([]lo.Tuple2[context.Context, T], count)
			size := 0
			index := 0

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if size < count {
							buffer[index] = lo.T2(ctx, value)
							size++
						} else {
							// Buffer is full, emit the oldest item
							destination.NextWithContext(buffer[index].A, buffer[index].B)
							buffer[index] = lo.T2(ctx, value)
						}
						index = (index + 1) % count
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SkipUntil suppresses items emitted by an Observable until a second Observable
// emits an item or completes. It will then emit all the subsequent items. If the
// second Observable is empty, SkipUntil will not emit any items. If the second
// Observable emits an item or completes, SkipUntil will emit all items.
func SkipUntil[T, S any](signal Observable[S]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			ready := uint32(0)

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							if atomic.LoadUint32(&ready) == 1 {
								destination.NextWithContext(ctx, value)
							}
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			subscriptions.AddUnsubscribable(
				signal.SubscribeWithContext(
					subscriberCtx,
					OnNextWithContext(
						func(ctx context.Context, value S) {
							atomic.StoreUint32(&ready, 1)
						},
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// Take emits only the first n items emitted by an Observable. If the count is
// greater than the number of items emitted by the source Observable, Take will
// emit all items. If the count is zero, Take will not emit any items.
func Take[T any](count int64) func(Observable[T]) Observable[T] {
	if count < 0 {
		panic(ErrTakeWrongCount)
	}

	return func(source Observable[T]) Observable[T] {
		if count == 0 {
			return Empty[T]() // Warning: the `source` will never be subscribed
		}

		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var index int64

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						destination.NextWithContext(ctx, value)

						index++

						if index >= count {
							destination.CompleteWithContext(ctx)
						}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// TakeWhile emits items emitted by an Observable so long as a specified condition
// is true. It will then complete. If the condition is never true, TakeWhile will
// not emit any items. If the condition is true on the first item, TakeWhile will
// emit all items. If the condition is false on the first item, TakeWhile will not
// emit any items.
func TakeWhile[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return TakeWhileIWithContext(func(ctx context.Context, v T, _ int64) (context.Context, bool) {
		return ctx, predicate(v)
	})
}

// TakeWhileWithContext emits items emitted by an Observable so long as a specified condition
// is true. It will then complete. If the condition is never true, TakeWhile will
// not emit any items. If the condition is true on the first item, TakeWhile will
// emit all items. If the condition is false on the first item, TakeWhile will not
// emit any items.
func TakeWhileWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return TakeWhileIWithContext(func(ctx context.Context, v T, _ int64) (context.Context, bool) {
		return predicate(ctx, v)
	})
}

// TakeWhileI emits items emitted by an Observable so long as a specified condition
// is true. It will then complete. If the condition is never true, TakeWhile will
// not emit any items. If the condition is true on the first item, TakeWhile will
// emit all items. If the condition is false on the first item, TakeWhile will not
// emit any items.
func TakeWhileI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return TakeWhileIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return ctx, predicate(v, i)
	})
}

// TakeWhileIWithContext emits items emitted by an Observable so long as a specified condition
// is true. It will then complete. If the condition is never true, TakeWhile will
// not emit any items. If the condition is true on the first item, TakeWhile will
// emit all items. If the condition is false on the first item, TakeWhile will not
// emit any items.
func TakeWhileIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			skipping := false
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if !skipping {
							if currentCtx, ok := predicate(ctx, value, i); ok {
								destination.NextWithContext(currentCtx, value)
							} else {
								destination.CompleteWithContext(currentCtx)
								skipping = true
							}
						}

						i++
					},
					func(ctx context.Context, err error) {
						if !skipping {
							destination.ErrorWithContext(ctx, err)
						}
					},
					func(ctx context.Context) {
						if !skipping {
							destination.CompleteWithContext(ctx)
						}
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// TakeLast emits only the last n items emitted by an Observable. If the count is
// greater than the number of items emitted by the source Observable, TakeLast will
// emit all items. If the count is zero, TakeLast will not emit any items.
func TakeLast[T any](count int) func(Observable[T]) Observable[T] {
	if count < 0 {
		panic(ErrTakeLastWrongCount)
	}

	return func(source Observable[T]) Observable[T] {
		if count == 0 {
			return Empty[T]() // Warning: the `source` will never be subscribed
		}

		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			// Use a circular buffer to avoid memory allocations
			buffer := make([]lo.Tuple2[context.Context, T], 0, count)
			index := 0

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if index >= count {
							buffer = buffer[1:]
						}

						buffer = append(buffer, lo.T2(ctx, value))
						index++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						// Emit items in order, starting from the oldest
						for i := 0; i < count && i < index; i++ {
							destination.NextWithContext(buffer[i].A, buffer[i].B)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// TakeUntil emits items emitted by an Observable until a second Observable emits
// an item or completes. It will then complete. If the second Observable is empty,
// TakeUntil will emit all items. If the second Observable emits an item or completes,
// TakeUntil will emit all items. If the second Observable emits an item or completes,
// TakeUntil will complete.
func TakeUntil[T, S any](signal Observable[S]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			ready := uint32(0)

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							if atomic.LoadUint32(&ready) == 1 {
								return
							}

							destination.NextWithContext(ctx, value)
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			subscriptions.AddUnsubscribable(
				signal.SubscribeWithContext(
					subscriberCtx,
					OnNextWithContext(
						func(ctx context.Context, value S) {
							atomic.StoreUint32(&ready, 1)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// Head emits only the first item emitted by an Observable. If the source Observable
// is empty, Head will emit an error.
func Head[T any]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						destination.NextWithContext(ctx, value)
						destination.CompleteWithContext(ctx)
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.ErrorWithContext(ctx, ErrHeadEmpty)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Tail emits only the last item emitted by an Observable. If the source Observable
// is empty, Tail will emit an error.
func Tail[T any]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var last lo.Tuple2[context.Context, T]

			hasValue := false

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						last = lo.T2(ctx, value)
						hasValue = true
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if hasValue {
							destination.NextWithContext(last.A, last.B)
							destination.CompleteWithContext(ctx)
						} else {
							destination.ErrorWithContext(ctx, ErrTailEmpty)
						}
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// First emits only the first item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, First will emit an error.
func First[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return FirstI(func(v T, _ int64) bool {
		return predicate(v)
	})
}

// FirstWithContext emits only the first item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, First will emit an error.
func FirstWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return FirstIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return predicate(ctx, v)
	})
}

// FirstI emits only the first item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, FirstI will emit an error.
func FirstI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return FirstIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return ctx, predicate(v, i)
	})
}

// FirstIWithContext emits only the first item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, FirstI will emit an error.
func FirstIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if currentCtx, ok := predicate(ctx, value, i); ok {
							destination.NextWithContext(currentCtx, value)
							destination.CompleteWithContext(currentCtx)
						}

						i++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.ErrorWithContext(ctx, ErrFirstEmpty)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Last emits only the last item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, Last will emit an error.
func Last[T any](predicate func(item T) bool) func(Observable[T]) Observable[T] {
	return LastI(func(v T, _ int64) bool {
		return predicate(v)
	})
}

// LastWithContext emits only the last item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, Last will emit an error.
func LastWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return LastIWithContext(func(ctx context.Context, item T, index int64) (context.Context, bool) {
		return predicate(ctx, item)
	})
}

// LastI emits only the last item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, LastI will emit an error.
func LastI[T any](predicate func(item T, index int64) bool) func(Observable[T]) Observable[T] {
	return LastIWithContext(func(ctx context.Context, v T, i int64) (context.Context, bool) {
		return ctx, predicate(v, i)
	})
}

// LastIWithContext emits only the last item emitted by an Observable that satisfies a specified
// condition. If the source Observable is empty, LastI will emit an error.
func LastIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var last lo.Tuple2[context.Context, T]

			hasValue := false
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if newCtx, ok := predicate(ctx, value, i); ok {
							last = lo.T2(newCtx, value)
							hasValue = true
						}

						i++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if hasValue {
							destination.NextWithContext(last.A, last.B)
							destination.CompleteWithContext(last.A)
						} else {
							destination.ErrorWithContext(ctx, ErrLastEmpty)
						}
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ElementAt emits only the nth item emitted by an Observable. If the source Observable
// emits fewer than n items, ElementAt will emit an error.
func ElementAt[T any](nth int) func(Observable[T]) Observable[T] {
	if nth < 0 {
		panic(ErrElementAtWrongNth)
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			count := 0

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if count == nth {
							destination.NextWithContext(ctx, value)
							destination.CompleteWithContext(ctx)
							return
						}

						count++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.ErrorWithContext(ctx, ErrElementAtNotFound)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ElementAtOrDefault emits only the nth item emitted by an Observable. If the source
// Observable emits fewer than n items, ElementAtOrDefault will emit a fallback value.
func ElementAtOrDefault[T any](nth int64, fallback T) func(Observable[T]) Observable[T] {
	if nth < 0 {
		panic(ErrElementAtOrDefaultWrongNth)
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			count := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if count == nth {
							destination.NextWithContext(ctx, value)
							destination.CompleteWithContext(ctx)
							return
						}

						count++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, fallback)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}
