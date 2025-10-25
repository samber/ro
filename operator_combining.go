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

//nolint:nestif,funlen,gocyclo
package ro

import (
	"context"
	"sync"
	"sync/atomic"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/xatomic"
)

// Merge creates an Observable that emits items from multiple source Observables, interleaved as they are emitted.
// Play: https://go.dev/play/p/Nerpzkth1lT
func Merge[T any](sources ...Observable[T]) Observable[T] {
	return MergeAll[T]()(Just(sources...))
}

// MergeWith merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith[T any](observables ...Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		list := make([]Observable[T], len(observables)+1)
		list[0] = obsA
		copy(list[1:], observables)
		return MergeAll[T]()(Just(list...))
	}
}

// MergeWith1 merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith1[T any](obsB Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		return MergeAll[T]()(Just(obsA, obsB))
	}
}

// MergeWith2 merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith2[T any](obsB, obsC Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		return MergeAll[T]()(Just(obsA, obsB, obsC))
	}
}

// MergeWith3 merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith3[T any](obsB, obsC, obsD Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		return MergeAll[T]()(Just(obsA, obsB, obsC, obsD))
	}
}

// MergeWith4 merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith4[T any](obsB, obsC, obsD, obsE Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		return MergeAll[T]()(Just(obsA, obsB, obsC, obsD, obsE))
	}
}

// MergeWith5 merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
//
// It is a curried function that takes the first Observable as an argument.
func MergeWith5[T any](obsB, obsC, obsD, obsE, obsF Observable[T]) func(Observable[T]) Observable[T] {
	return func(obsA Observable[T]) Observable[T] {
		return MergeAll[T]()(Just(obsA, obsB, obsC, obsD, obsE, obsF))
	}
}

// MergeAll converts a higher-order Observable into a first-order Observable which
// concurrently delivers all values that are emitted on the inner Observables.
// It subscribes to each inner Observable as they arrive, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
func MergeAll[T any]() func(Observable[Observable[T]]) Observable[T] {
	return func(sources Observable[Observable[T]]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var parentCtx context.Context
			var parentCtxMu sync.Mutex // atomic.Value has been introduced in go 1.19 and this library support go 1.18

			subscriptions := NewSubscription(nil)

			// default value is not 0, because it counts the outer Observable `sources`
			subscriptionsCount := int32(1)

			onDone := func() {
				newCount := atomic.AddInt32(&subscriptionsCount, -1)

				// when equal to 0, it means both the outer and inner Observables are done
				if newCount == 0 {
					parentCtxMu.Lock()
					destination.CompleteWithContext(parentCtx)
					parentCtxMu.Unlock()
				}
			}

			subscriptions.AddUnsubscribable(
				sources.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, source Observable[T]) {
							atomic.AddInt32(&subscriptionsCount, 1)

							subscriptions.AddUnsubscribable(
								source.SubscribeWithContext(
									ctx,
									NewObserverWithContext(
										destination.NextWithContext,
										destination.ErrorWithContext,
										func(ctx context.Context) {
											onDone()
										},
									),
								),
							)
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							parentCtxMu.Lock()
							parentCtx = ctx
							parentCtxMu.Unlock()

							onDone()
						},
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// MergeMap applies a projection function to each item emitted by the source
// Observable and then merges the results into a single Observable.
func MergeMap[T, R any](projection func(item T) Observable[R]) func(Observable[T]) Observable[R] {
	return MergeMapIWithContext(func(ctx context.Context, item T, index int64) (context.Context, Observable[R]) {
		return ctx, projection(item)
	})
}

// MergeMapWithContext applies a projection function to each item emitted by the source
// Observable and then merges the results into a single Observable.
func MergeMapWithContext[T, R any](projection func(ctx context.Context, item T) Observable[R]) func(Observable[T]) Observable[R] {
	return MergeMapIWithContext(func(ctx context.Context, item T, _ int64) (context.Context, Observable[R]) {
		return ctx, projection(ctx, item)
	})
}

// MergeMapI applies a projection function to each item emitted by the source
// Observable and then merges the results into a single Observable.
func MergeMapI[T, R any](projection func(item T, index int64) Observable[R]) func(Observable[T]) Observable[R] {
	return MergeMapIWithContext(func(ctx context.Context, item T, index int64) (context.Context, Observable[R]) {
		return ctx, projection(item, index)
	})
}

// MergeMapIWithContext applies a projection function to each item emitted by the source
// Observable and then merges the results into a single Observable.
func MergeMapIWithContext[T, R any](projection func(ctx context.Context, item T, index int64) (context.Context, Observable[R])) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		i := int64(0)

		return MergeAll[R]()(
			NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[Observable[R]]) Teardown {
				sub := source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							destination.NextWithContext(projection(ctx, value, i))

							i++
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				)

				return sub.Unsubscribe
			}),
		)
	}
}

// CombineLatestWith combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func CombineLatestWith[A, B any](obsB Observable[B]) func(Observable[A]) Observable[lo.Tuple2[A, B]] {
	return CombineLatestWith1[A](obsB)
}

// CombineLatestWith1 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func CombineLatestWith1[A, B any](obsB Observable[B]) func(Observable[A]) Observable[lo.Tuple2[A, B]] {
	return func(obsA Observable[A]) Observable[lo.Tuple2[A, B]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple2[A, B]]) Teardown {
			var valueA xatomic.Pointer[A]
			var valueB xatomic.Pointer[B]

			// 0: not done
			// 1: partially done
			// 2: done
			// 3: error
			var status int32

			onUpdate := func(ctx context.Context, a *A, b *B) {
				if atomic.LoadInt32(&status) < 2 {
					if a == nil {
						a = valueA.Load()
					}

					if b == nil {
						b = valueB.Load()
					}

					if a != nil && b != nil {
						destination.NextWithContext(ctx, lo.T2(*a, *b))
					}
				}
			}

			onCompleted := func(ctx context.Context) {
				if atomic.LoadInt32(&status) == 2 {
					destination.CompleteWithContext(ctx)
				}
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				obsA.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v A) {
							valueA.Store(&v)
							onUpdate(ctx, &v, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 3)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsB.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v B) {
							valueB.Store(&v)
							onUpdate(ctx, nil, &v)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 3)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			return func() {
				atomic.StoreInt32(&status, 2)
				subscriptions.Unsubscribe()
			}
		})
	}
}

// CombineLatestWith2 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func CombineLatestWith2[A, B, C any](obsB Observable[B], obsC Observable[C]) func(Observable[A]) Observable[lo.Tuple3[A, B, C]] {
	return func(obsA Observable[A]) Observable[lo.Tuple3[A, B, C]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple3[A, B, C]]) Teardown {
			var valueA xatomic.Pointer[A]
			var valueB xatomic.Pointer[B]
			var valueC xatomic.Pointer[C]

			// 0: not done
			// 1: partially done
			// 2: partially done
			// 3: done
			// 4: error
			var status int32

			onUpdate := func(ctx context.Context, a *A, b *B, c *C) {
				if atomic.LoadInt32(&status) < 3 {
					if a == nil {
						a = valueA.Load()
					}

					if b == nil {
						b = valueB.Load()
					}

					if c == nil {
						c = valueC.Load()
					}

					if a != nil && b != nil && c != nil {
						destination.NextWithContext(ctx, lo.T3(*a, *b, *c))
					}
				}
			}

			onCompleted := func(ctx context.Context) {
				if atomic.LoadInt32(&status) == 3 {
					destination.CompleteWithContext(ctx)
				}
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				obsA.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v A) {
							valueA.Store(&v)
							onUpdate(ctx, &v, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 4)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsB.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v B) {
							valueB.Store(&v)
							onUpdate(ctx, nil, &v, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 4)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsC.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v C) {
							valueC.Store(&v)
							onUpdate(ctx, nil, nil, &v)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 4)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			return func() {
				atomic.StoreInt32(&status, 3)
				subscriptions.Unsubscribe()
			}
		})
	}
}

// CombineLatestWith3 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func CombineLatestWith3[A, B, C, D any](obsB Observable[B], obsC Observable[C], obsD Observable[D]) func(Observable[A]) Observable[lo.Tuple4[A, B, C, D]] {
	return func(obsA Observable[A]) Observable[lo.Tuple4[A, B, C, D]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple4[A, B, C, D]]) Teardown {
			var valueA xatomic.Pointer[A]
			var valueB xatomic.Pointer[B]
			var valueC xatomic.Pointer[C]
			var valueD xatomic.Pointer[D]

			// 0: not done
			// 1: partially done
			// 2: partially done
			// 3: partially done
			// 4: done
			// 5: error
			var status int32

			onUpdate := func(ctx context.Context, a *A, b *B, c *C, d *D) {
				if atomic.LoadInt32(&status) < 4 {
					if a == nil {
						a = valueA.Load()
					}

					if b == nil {
						b = valueB.Load()
					}

					if c == nil {
						c = valueC.Load()
					}

					if d == nil {
						d = valueD.Load()
					}

					if a != nil && b != nil && c != nil && d != nil {
						destination.NextWithContext(ctx, lo.T4(*a, *b, *c, *d))
					}
				}
			}

			onCompleted := func(ctx context.Context) {
				if atomic.LoadInt32(&status) == 4 {
					destination.CompleteWithContext(ctx)
				}
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				obsA.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v A) {
							valueA.Store(&v)
							onUpdate(ctx, &v, nil, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 5)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsB.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v B) {
							valueB.Store(&v)
							onUpdate(ctx, nil, &v, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 5)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsC.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v C) {
							valueC.Store(&v)
							onUpdate(ctx, nil, nil, &v, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 5)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsD.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v D) {
							valueD.Store(&v)
							onUpdate(ctx, nil, nil, nil, &v)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 5)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			return func() {
				atomic.StoreInt32(&status, 4)
				subscriptions.Unsubscribe()
			}
		})
	}
}

// CombineLatestWith4 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func CombineLatestWith4[A, B, C, D, E any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E]) func(Observable[A]) Observable[lo.Tuple5[A, B, C, D, E]] {
	return func(obsA Observable[A]) Observable[lo.Tuple5[A, B, C, D, E]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple5[A, B, C, D, E]]) Teardown {
			var valueA xatomic.Pointer[A]
			var valueB xatomic.Pointer[B]
			var valueC xatomic.Pointer[C]
			var valueD xatomic.Pointer[D]
			var valueE xatomic.Pointer[E]

			// 0: not done
			// 1: partially done
			// 2: partially done
			// 3: partially done
			// 4: partially done
			// 5: done
			// 6: error
			var status int32

			onUpdate := func(ctx context.Context, a *A, b *B, c *C, d *D, e *E) {
				if atomic.LoadInt32(&status) < 5 {
					if a == nil {
						a = valueA.Load()
					}

					if b == nil {
						b = valueB.Load()
					}

					if c == nil {
						c = valueC.Load()
					}

					if d == nil {
						d = valueD.Load()
					}

					if e == nil {
						e = valueE.Load()
					}

					if a != nil && b != nil && c != nil && d != nil && e != nil {
						destination.NextWithContext(ctx, lo.T5(*a, *b, *c, *d, *e))
					}
				}
			}

			onCompleted := func(ctx context.Context) {
				if atomic.LoadInt32(&status) == 5 {
					destination.CompleteWithContext(ctx)
				}
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				obsA.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v A) {
							valueA.Store(&v)
							onUpdate(ctx, &v, nil, nil, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 6)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsB.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v B) {
							valueB.Store(&v)
							onUpdate(ctx, nil, &v, nil, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 6)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsC.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v C) {
							valueC.Store(&v)
							onUpdate(ctx, nil, nil, &v, nil, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 6)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsD.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v D) {
							valueD.Store(&v)
							onUpdate(ctx, nil, nil, nil, &v, nil)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 6)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				obsE.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v E) {
							valueE.Store(&v)
							onUpdate(ctx, nil, nil, nil, nil, &v)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, 6)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							atomic.AddInt32(&status, 1)
							onCompleted(ctx)
						},
					),
				),
			)

			return func() {
				atomic.StoreInt32(&status, 5)
				subscriptions.Unsubscribe()
			}
		})
	}
}

// CombineLatestAll combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
func CombineLatestAll[T any]() func(Observable[Observable[T]]) Observable[[]T] {
	return func(sources Observable[Observable[T]]) Observable[[]T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			subscriptions := NewSubscription(nil)

			var observables []Observable[T]

			var values []*xatomic.Pointer[T]

			// -1: error
			// 0: done
			// 1: partially done
			// 2: partially done
			// .: partially done
			// .: partially done
			// n: not done
			var status int32

			onUpdate := func(ctx context.Context) {
				if atomic.LoadInt32(&status) > 0 {
					result := make([]T, len(values))

					for i := range values {
						v := values[i].Load()
						if v == nil {
							return
						}

						result[i] = *v
					}

					destination.NextWithContext(ctx, result)
				}
			}

			onCompleted := func(ctx context.Context) {
				if atomic.LoadInt32(&status) == 0 {
					destination.CompleteWithContext(ctx)
				}
			}

			subscribeInner := func() {
				// init
				atomic.StoreInt32(&status, int32(len(observables)))

				values = make([]*xatomic.Pointer[T], len(observables))
				for i := range observables {
					values[i] = new(xatomic.Pointer[T])
				}

				// inner subscriptions
				for i := range observables {
					j := i

					subscriptions.AddUnsubscribable(
						observables[j].SubscribeWithContext(
							subscriberCtx,
							NewObserverWithContext(
								func(ctx context.Context, v T) {
									values[j].Store(&v)
									onUpdate(ctx)
								},
								func(ctx context.Context, err error) {
									atomic.StoreInt32(&status, -1)
									destination.ErrorWithContext(ctx, err)
								},
								func(ctx context.Context) {
									atomic.AddInt32(&status, -1)
									onCompleted(ctx)
								},
							),
						),
					)
				}
			}

			// outer subscription
			subscriptions.AddUnsubscribable(
				sources.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, v Observable[T]) {
							observables = append(observables, v)
						},
						func(ctx context.Context, err error) {
							atomic.StoreInt32(&status, -1)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							if len(observables) > 0 {
								subscribeInner()
							} else {
								atomic.StoreInt32(&status, 0)
								destination.CompleteWithContext(ctx)
							}
						},
					),
				),
			)

			return func() {
				atomic.StoreInt32(&status, 0)
				subscriptions.Unsubscribe()
			}
		})
	}
}

// CombineLatestAllAny combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
func CombineLatestAllAny() func(Observable[Observable[any]]) Observable[[]any] {
	return CombineLatestAll[any]()
}

// ConcatWith concatenates the source Observable with other Observables. It subscribes
// to each inner Observable only after the previous one completes, maintaining their
// order. It completes when all inner Observables are done.
//
// It is a curried function that takes the other Observables as arguments.
func ConcatWith[T any](obs ...Observable[T]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return ConcatAll[T]()(Just(append([]Observable[T]{source}, obs...)...))
	}
}

// ConcatAll concatenates the source Observable with other Observables. It subscribes
// to each inner Observable only after the previous one completes, maintaining their
// order. It completes when all inner Observables are done.
func ConcatAll[T any]() func(Observable[Observable[T]]) Observable[T] {
	return func(sources Observable[Observable[T]]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				sources.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, source Observable[T]) {
							sub := source.SubscribeWithContext(
								ctx,
								NewObserverWithContext(
									destination.NextWithContext,
									func(ctx context.Context, err error) {
										subscriptions.Unsubscribe()
										destination.ErrorWithContext(ctx, err)
									},
									func(ctx context.Context) {},
								),
							)

							// `subscriptions` cancels `sub` when it unsubscribes
							// but `sub` cannot unsubscribe `subscriptions`
							subscriptions.AddUnsubscribable(sub)
							sub.Wait()
						},
						func(ctx context.Context, err error) {
							subscriptions.Unsubscribe()
							destination.ErrorWithContext(ctx, err)
						},
						destination.CompleteWithContext,
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// StartWith emits the given values before emitting the values from the source Observable.
func StartWith[T any](prefixes ...T) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			for i := range prefixes {
				destination.NextWithContext(subscriberCtx, prefixes[i])
			}

			sub := source.SubscribeWithContext(subscriberCtx, destination)

			return sub.Unsubscribe
		})
	}
}

// EndWith emits the given values after emitting the values from the source Observable.
func EndWith[T any](suffixes ...T) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					destination.NextWithContext,
					destination.ErrorWithContext,
					func(ctx context.Context) {
						for i := range suffixes {
							destination.NextWithContext(ctx, suffixes[i])
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Pairwise emits the previous and current values as a pair of two values.
// Play: https://go.dev/play/p/0YujgFTL4e0
func Pairwise[T any]() func(Observable[T]) Observable[[]T] {
	return func(source Observable[T]) Observable[[]T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			count := int64(0)

			var last T

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if count > 0 {
							destination.NextWithContext(ctx, []T{last, value})
						}

						count++
						last = value
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// RaceWith creates an Observable that mirrors the first source Observable to
// emit a next, error or complete notification from the combination of the
// Observable to which the operator is applied and supplied Observables. It
// cancels the subscriptions to all other Observables. It completes when the
// source Observable completes. If the source Observable errors, it errors with
// the same error.
//
// It is a curried function that takes the other Observables as arguments.
func RaceWith[T any](sources ...Observable[T]) func(Observable[T]) Observable[T] {
	if len(sources) == 0 {
		return func(source Observable[T]) Observable[T] {
			return source
		}
	}

	return func(source Observable[T]) Observable[T] {
		all := append([]Observable[T]{source}, sources...)

		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			subscriptions := make([]Subscription, len(all))
			won := int32(-1)
			mu := sync.Mutex{}

			unsubscribeOthers := func(except int) {
				mu.Lock()
				defer mu.Unlock()

				unsubscription := NewSubscription(nil)

				for i := range subscriptions {
					if i != except && subscriptions[i] != nil {
						unsubscription.AddUnsubscribable(subscriptions[i])
					}
				}

				unsubscription.Unsubscribe()
			}

			for i := range all {
				j := i

				if atomic.LoadInt32(&won) != -1 {
					continue
				}

				sub := all[j].SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							if atomic.CompareAndSwapInt32(&won, -1, int32(j)) || atomic.LoadInt32(&won) == int32(j) {
								destination.NextWithContext(ctx, value)
								unsubscribeOthers(j)
							}
						},
						func(ctx context.Context, err error) {
							if atomic.CompareAndSwapInt32(&won, -1, int32(j)) || atomic.LoadInt32(&won) == int32(j) {
								destination.ErrorWithContext(ctx, err)
								unsubscribeOthers(j)
							}
						},
						func(ctx context.Context) {
							if atomic.CompareAndSwapInt32(&won, -1, int32(j)) || atomic.LoadInt32(&won) == int32(j) {
								destination.CompleteWithContext(ctx)
								unsubscribeOthers(j)
							}
						},
					),
				)

				// Check if a winner was determined during subscription
				winner := atomic.LoadInt32(&won)
				hasWinner := winner != -1
				isWinner := hasWinner && winner == int32(j)

				mu.Lock()
				if !hasWinner {
					// No winner yet, store the subscription
					subscriptions[j] = sub
				} else if !isWinner {
					// Another source won, unsubscribe this one
					sub.Unsubscribe()
				}
				// If this source won, keep the subscription active
				mu.Unlock()
			}

			return func() {
				unsubscribeOthers(-1)
			}
		})
	}
}

type zipDestination interface {
	ErrorWithContext(context.Context, error)
	CompleteWithContext(context.Context)
}

// This code is dity but much more concise than the original implementation.
func zipInnerSubscription[T any](subscriberCtx context.Context, obs Observable[T], mu *sync.Mutex, values *[]*T, completed *bool, onUpdate func(context.Context), destination zipDestination, subscriptions Subscription) {
	subscriptions.AddUnsubscribable(
		obs.SubscribeWithContext(
			subscriberCtx,
			NewObserverWithContext(
				func(ctx context.Context, v T) {
					mu.Lock()

					*values = append(*values, &v)

					mu.Unlock()

					onUpdate(ctx)
				},
				func(ctx context.Context, err error) {
					mu.Lock()

					*completed = true

					mu.Unlock()

					destination.ErrorWithContext(ctx, err)
					subscriptions.Unsubscribe()
				},
				func(ctx context.Context) {
					mu.Lock()

					*completed = true

					if len(*values) == 0 {
						mu.Unlock()
						destination.CompleteWithContext(ctx)
					} else {
						mu.Unlock()
					}

					subscriptions.Unsubscribe()
				},
			),
		),
	)
}

// ZipWith combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith[A, B any](obsB Observable[B]) func(Observable[A]) Observable[lo.Tuple2[A, B]] {
	return ZipWith1[A](obsB)
}

// ZipWith1 combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith1[A, B any](obsB Observable[B]) func(Observable[A]) Observable[lo.Tuple2[A, B]] {
	return func(obsA Observable[A]) Observable[lo.Tuple2[A, B]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple2[A, B]]) Teardown {
			var mu sync.Mutex

			var valueA []*A
			var valueB []*B

			var completedA bool
			var completedB bool

			onUpdate := func(ctx context.Context) {
				mu.Lock()

				if len(valueA) > 0 && len(valueB) > 0 {
					a := valueA[0]
					b := valueB[0]
					valueA = valueA[1:]
					valueB = valueB[1:]

					mu.Unlock() // unlock before calling destination.Next to prevent long locks

					destination.NextWithContext(ctx, lo.T2(*a, *b)) // @TODO: Send the last context ?

					mu.Lock()

					if (completedA && len(valueA) == 0) ||
						(completedB && len(valueB) == 0) {
						destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					}
				}

				mu.Unlock()
			}

			subscriptions := NewSubscription(nil)
			zipInnerSubscription(subscriberCtx, obsA, &mu, &valueA, &completedA, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsB, &mu, &valueB, &completedB, onUpdate, destination, subscriptions)

			return func() {
				subscriptions.Unsubscribe()

				// free memory
				mu.Lock()

				completedA = true
				completedB = true
				valueA = nil
				valueB = nil

				mu.Unlock()
			}
		})
	}
}

// ZipWith2 combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith2[A, B, C any](obsB Observable[B], obsC Observable[C]) func(Observable[A]) Observable[lo.Tuple3[A, B, C]] {
	return func(obsA Observable[A]) Observable[lo.Tuple3[A, B, C]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple3[A, B, C]]) Teardown {
			var mu sync.Mutex

			var valueA []*A
			var valueB []*B
			var valueC []*C

			var completedA bool
			var completedB bool
			var completedC bool

			onUpdate := func(ctx context.Context) {
				mu.Lock()

				if len(valueA) > 0 && len(valueB) > 0 && len(valueC) > 0 {
					a := valueA[0]
					b := valueB[0]
					c := valueC[0]
					valueA = valueA[1:]
					valueB = valueB[1:]
					valueC = valueC[1:]

					mu.Unlock() // unlock before calling destination.Next to prevent long locks

					destination.NextWithContext(ctx, lo.T3(*a, *b, *c)) // @TODO: Send the last context ?

					mu.Lock()

					if (completedA && len(valueA) == 0) ||
						(completedB && len(valueB) == 0) ||
						(completedC && len(valueC) == 0) {
						destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					}
				}

				mu.Unlock()
			}

			subscriptions := NewSubscription(nil)
			zipInnerSubscription(subscriberCtx, obsA, &mu, &valueA, &completedA, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsB, &mu, &valueB, &completedB, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsC, &mu, &valueC, &completedC, onUpdate, destination, subscriptions)

			return func() {
				subscriptions.Unsubscribe()

				// free memory
				mu.Lock()

				completedA = true
				completedB = true
				completedC = true
				valueA = nil
				valueB = nil
				valueC = nil

				mu.Unlock()
			}
		})
	}
}

// ZipWith3 combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith3[A, B, C, D any](obsB Observable[B], obsC Observable[C], obsD Observable[D]) func(Observable[A]) Observable[lo.Tuple4[A, B, C, D]] {
	return func(obsA Observable[A]) Observable[lo.Tuple4[A, B, C, D]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple4[A, B, C, D]]) Teardown {
			var mu sync.Mutex

			var valueA []*A
			var valueB []*B
			var valueC []*C
			var valueD []*D

			var completedA bool
			var completedB bool
			var completedC bool
			var completedD bool

			onUpdate := func(ctx context.Context) {
				mu.Lock()

				if len(valueA) > 0 && len(valueB) > 0 && len(valueC) > 0 && len(valueD) > 0 {
					a := valueA[0]
					b := valueB[0]
					c := valueC[0]
					d := valueD[0]
					valueA = valueA[1:]
					valueB = valueB[1:]
					valueC = valueC[1:]
					valueD = valueD[1:]

					mu.Unlock() // unlock before calling destination.Next to prevent long locks

					destination.NextWithContext(ctx, lo.T4(*a, *b, *c, *d)) // @TODO: Send the last context ?

					mu.Lock()

					if (completedA && len(valueA) == 0) ||
						(completedB && len(valueB) == 0) ||
						(completedC && len(valueC) == 0) ||
						(completedD && len(valueD) == 0) {
						destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					}
				}

				mu.Unlock()
			}

			subscriptions := NewSubscription(nil)
			zipInnerSubscription(subscriberCtx, obsA, &mu, &valueA, &completedA, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsB, &mu, &valueB, &completedB, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsC, &mu, &valueC, &completedC, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsD, &mu, &valueD, &completedD, onUpdate, destination, subscriptions)

			return func() {
				subscriptions.Unsubscribe()

				// free memory
				mu.Lock()

				completedA = true
				completedB = true
				completedC = true
				completedD = true
				valueA = nil
				valueB = nil
				valueC = nil
				valueD = nil

				mu.Unlock()
			}
		})
	}
}

// ZipWith4 combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith4[A, B, C, D, E any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E]) func(Observable[A]) Observable[lo.Tuple5[A, B, C, D, E]] {
	return func(obsA Observable[A]) Observable[lo.Tuple5[A, B, C, D, E]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple5[A, B, C, D, E]]) Teardown {
			var mu sync.Mutex

			var valueA []*A
			var valueB []*B
			var valueC []*C
			var valueD []*D
			var valueE []*E

			var completedA bool
			var completedB bool
			var completedC bool
			var completedD bool
			var completedE bool

			onUpdate := func(ctx context.Context) {
				mu.Lock()

				if len(valueA) > 0 && len(valueB) > 0 && len(valueC) > 0 && len(valueD) > 0 && len(valueE) > 0 {
					a := valueA[0]
					b := valueB[0]
					c := valueC[0]
					d := valueD[0]
					e := valueE[0]
					valueA = valueA[1:]
					valueB = valueB[1:]
					valueC = valueC[1:]
					valueD = valueD[1:]
					valueE = valueE[1:]

					mu.Unlock() // unlock before calling destination.Next to prevent long locks

					destination.NextWithContext(ctx, lo.T5(*a, *b, *c, *d, *e)) // @TODO: Send the last context ?

					mu.Lock()

					if (completedA && len(valueA) == 0) ||
						(completedB && len(valueB) == 0) ||
						(completedC && len(valueC) == 0) ||
						(completedD && len(valueD) == 0) ||
						(completedE && len(valueE) == 0) {
						destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					}
				}

				mu.Unlock()
			}

			subscriptions := NewSubscription(nil)
			zipInnerSubscription(subscriberCtx, obsA, &mu, &valueA, &completedA, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsB, &mu, &valueB, &completedB, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsC, &mu, &valueC, &completedC, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsD, &mu, &valueD, &completedD, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsE, &mu, &valueE, &completedE, onUpdate, destination, subscriptions)

			return func() {
				subscriptions.Unsubscribe()

				// free memory
				mu.Lock()

				completedA = true
				completedB = true
				completedC = true
				completedD = true
				completedE = true
				valueA = nil
				valueB = nil
				valueC = nil
				valueD = nil
				valueE = nil

				mu.Unlock()
			}
		})
	}
}

// ZipWith5 combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
//
// It is a curried function that takes the other Observable as an argument.
func ZipWith5[A, B, C, D, E, F any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E], obsF Observable[F]) func(Observable[A]) Observable[lo.Tuple6[A, B, C, D, E, F]] {
	return func(obsA Observable[A]) Observable[lo.Tuple6[A, B, C, D, E, F]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[lo.Tuple6[A, B, C, D, E, F]]) Teardown {
			var mu sync.Mutex

			var valueA []*A
			var valueB []*B
			var valueC []*C
			var valueD []*D
			var valueE []*E
			var valueF []*F

			var completedA bool
			var completedB bool
			var completedC bool
			var completedD bool
			var completedE bool
			var completedF bool

			onUpdate := func(ctx context.Context) {
				mu.Lock()

				if len(valueA) > 0 && len(valueB) > 0 && len(valueC) > 0 && len(valueD) > 0 && len(valueE) > 0 && len(valueF) > 0 {
					a := valueA[0]
					b := valueB[0]
					c := valueC[0]
					d := valueD[0]
					e := valueE[0]
					f := valueF[0]
					valueA = valueA[1:]
					valueB = valueB[1:]
					valueC = valueC[1:]
					valueD = valueD[1:]
					valueE = valueE[1:]
					valueF = valueF[1:]

					mu.Unlock() // unlock before calling destination.Next to prevent long locks

					destination.NextWithContext(ctx, lo.T6(*a, *b, *c, *d, *e, *f)) // @TODO: Send the last context ?

					mu.Lock()

					if (completedA && len(valueA) == 0) ||
						(completedB && len(valueB) == 0) ||
						(completedC && len(valueC) == 0) ||
						(completedD && len(valueD) == 0) ||
						(completedE && len(valueE) == 0) ||
						(completedF && len(valueF) == 0) {
						destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					}
				}

				mu.Unlock()
			}

			subscriptions := NewSubscription(nil)
			zipInnerSubscription(subscriberCtx, obsA, &mu, &valueA, &completedA, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsB, &mu, &valueB, &completedB, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsC, &mu, &valueC, &completedC, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsD, &mu, &valueD, &completedD, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsE, &mu, &valueE, &completedE, onUpdate, destination, subscriptions)
			zipInnerSubscription(subscriberCtx, obsF, &mu, &valueF, &completedF, onUpdate, destination, subscriptions)

			return func() {
				subscriptions.Unsubscribe()

				// free memory
				mu.Lock()

				completedA = true
				completedB = true
				completedC = true
				completedD = true
				completedE = true
				completedF = true
				valueA = nil
				valueB = nil
				valueC = nil
				valueD = nil
				valueE = nil
				valueF = nil

				mu.Unlock()
			}
		})
	}
}

func zipAllInnerSubscriptions[T any](outerCtx context.Context, sources []Observable[T], destination Observer[[]T]) Teardown {
	var mu sync.Mutex

	values := make([][]*T, len(sources))
	completed := make([]bool, len(sources))

	onUpdate := func(ctx context.Context) {
		mu.Lock()

		hasEmptyQueue := false

		for i := range sources {
			if len(values[i]) == 0 {
				hasEmptyQueue = true
				break
			}
		}

		if !hasEmptyQueue {
			result := make([]T, len(sources))
			for i := range sources {
				result[i] = *values[i][0]
				values[i] = values[i][1:]
			}

			mu.Unlock() // unlock before calling destination.Next to prevent long locks

			destination.NextWithContext(ctx, result) // @TODO: Send the last context ?

			mu.Lock()

			for i := range sources {
				if completed[i] && len(values[i]) == 0 {
					destination.CompleteWithContext(ctx) // @TODO: Send the last context ?
					break
				}
			}
		}

		mu.Unlock()
	}

	subscriptions := NewSubscription(nil)

	for i := range sources {
		j := i
		zipInnerSubscription(outerCtx, sources[i], &mu, &(values[j]), &(completed[j]), onUpdate, destination, subscriptions)
	}

	return func() {
		subscriptions.Unsubscribe()

		// free memory
		mu.Lock()

		completed = nil
		values = nil

		mu.Unlock()
	}
}

// ZipAll combines the values from the source Observable with the latest values
// from the other Observables. It emits only when all Observables have emitted
// at least one value. It completes when the source Observable completes.
func ZipAll[T any]() func(Observable[Observable[T]]) Observable[[]T] {
	return func(sources Observable[Observable[T]]) Observable[[]T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			innerSub := NewSubscription(nil)

			// First, we consume the high-order observable...
			outerSub := ToSlice[Observable[T]]()(sources).
				SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, flattenSources []Observable[T]) {
							innerSub.Add(
								// ...then we zip all inner observables.
								zipAllInnerSubscriptions(ctx, flattenSources, destination),
							)
						},
						func(ctx context.Context, err error) {
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							destination.CompleteWithContext(ctx)
						},
					),
				)

			return func() {
				outerSub.Unsubscribe()
				innerSub.Unsubscribe()
			}
		})
	}
}
