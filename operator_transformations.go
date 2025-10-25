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
	"sync"
	"sync/atomic"
	"time"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/xsync"
	"github.com/samber/ro/internal/xtime"
)

// Map applies a given project function to each item emitted by an Observable and emits the result.
func Map[T, R any](project func(item T) R) func(Observable[T]) Observable[R] {
	return MapIWithContext(func(ctx context.Context, v T, _ int64) (context.Context, R) {
		return ctx, project(v)
	})
}

// MapWithContext applies a given project function to each item emitted by an Observable and emits the result.
func MapWithContext[T, R any](project func(ctx context.Context, item T) (context.Context, R)) func(Observable[T]) Observable[R] {
	return MapIWithContext(func(ctx context.Context, v T, _ int64) (context.Context, R) {
		return project(ctx, v)
	})
}

// MapI applies a given project function to each item emitted by an Observable and emits the result.
func MapI[T, R any](project func(item T, index int64) R) func(Observable[T]) Observable[R] {
	return MapIWithContext(func(ctx context.Context, v T, i int64) (context.Context, R) {
		return ctx, project(v, i)
	})
}

// MapIWithContext applies a given project function to each item emitted by an Observable and emits the result.
func MapIWithContext[T, R any](project func(ctx context.Context, item T, index int64) (context.Context, R)) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						newCtx, result := project(ctx, value, i)
						destination.NextWithContext(newCtx, result)

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

// MapTo emits a constant value for each item emitted by an Observable.
func MapTo[T, R any](output R) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						// ignore value
						destination.NextWithContext(ctx, output)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MapErr applies a given project function to each item emitted by an Observable and emits the result.
func MapErr[T, R any](project func(item T) (R, error)) func(Observable[T]) Observable[R] {
	return MapErrIWithContext(func(ctx context.Context, t T, _ int64) (R, context.Context, error) {
		r, err := project(t)
		return r, ctx, err
	})
}

// MapErrWithContext applies a given project function to each item emitted by an Observable and emits the result.
func MapErrWithContext[T, R any](project func(ctx context.Context, item T) (R, context.Context, error)) func(Observable[T]) Observable[R] {
	return MapErrIWithContext(func(ctx context.Context, t T, _ int64) (R, context.Context, error) {
		return project(ctx, t)
	})
}

// MapErrI applies a given project function to each item emitted by an Observable and emits the result.
func MapErrI[T, R any](project func(item T, index int64) (R, error)) func(Observable[T]) Observable[R] {
	return MapErrIWithContext(func(ctx context.Context, v T, i int64) (R, context.Context, error) {
		r, err := project(v, i)
		return r, ctx, err
	})
}

// MapErrIWithContext applies a given project function to each item emitted by an Observable and emits the result.
func MapErrIWithContext[T, R any](project func(ctx context.Context, item T, index int64) (R, context.Context, error)) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
			count := int64(0)
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, t T) {
						v, ctx, err := project(ctx, t, count)
						count++

						if err != nil {
							destination.ErrorWithContext(ctx, err)
							return
						}

						destination.NextWithContext(ctx, v)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// FlatMap transforms the items emitted by an Observable into Observables,
// then flatten the emissions from those into a single Observable.
func FlatMap[T, R any](project func(item T) Observable[R]) func(Observable[T]) Observable[R] {
	return FlatMapI(func(v T, _ int64) Observable[R] {
		return project(v)
	})
}

// FlatMapWithContext transforms the items emitted by an Observable into Observables,
// then flatten the emissions from those into a single Observable.
func FlatMapWithContext[T, R any](project func(ctx context.Context, item T) Observable[R]) func(Observable[T]) Observable[R] {
	return FlatMapIWithContext(func(ctx context.Context, v T, _ int64) Observable[R] {
		return project(ctx, v)
	})
}

// FlatMapI transforms the items emitted by an Observable into Observables,
// then flatten the emissions from those into a single Observable.
func FlatMapI[T, R any](project func(item T, index int64) Observable[R]) func(Observable[T]) Observable[R] {
	return FlatMapIWithContext(func(ctx context.Context, v T, i int64) Observable[R] {
		return project(v, i)
	})
}

// FlatMapIWithContext transforms the items emitted by an Observable into Observables,
// then flatten the emissions from those into a single Observable.
func FlatMapIWithContext[T, R any](project func(ctx context.Context, item T, index int64) Observable[R]) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return ConcatAll[R]()(
			NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[Observable[R]]) Teardown {
				i := int64(0)

				sub := source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							destination.NextWithContext(ctx, project(ctx, value, i))

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

// Flatten flattens an Observable of Observables into a single Observable.
func Flatten[T any]() func(Observable[[]T]) Observable[T] {
	return func(source Observable[[]T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value []T) {
						for _, v := range value {
							destination.NextWithContext(ctx, v)
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

// Cast converts each value emitted by an Observable into a specified type.
func Cast[T, U any]() func(Observable[T]) Observable[U] {
	return func(source Observable[T]) Observable[U] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[U]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if v, ok := any(value).(U); ok {
							destination.NextWithContext(ctx, v)
						} else {
							destination.ErrorWithContext(ctx, newCastError[T, U]())
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

// Scan applies an accumulator function over an Observable and emits each intermediate result.
// Play: https://go.dev/play/p/jZD5FyPN3P_D
func Scan[T, R any](reduce func(accumulator R, item T) R, seed R) func(Observable[T]) Observable[R] {
	return ScanIWithContext(func(ctx context.Context, accumulator R, item T, _ int64) (context.Context, R) {
		return ctx, reduce(accumulator, item)
	}, seed)
}

// ScanWithContext applies an accumulator function over an Observable and emits each intermediate result.
func ScanWithContext[T, R any](reduce func(ctx context.Context, accumulator R, item T) (context.Context, R), seed R) func(Observable[T]) Observable[R] {
	return ScanIWithContext(func(ctx context.Context, accumulator R, item T, _ int64) (context.Context, R) {
		return reduce(ctx, accumulator, item)
	}, seed)
}

// ScanI applies an accumulator function over an Observable and emits each intermediate result.
func ScanI[T, R any](reduce func(accumulator R, item T, index int64) R, seed R) func(Observable[T]) Observable[R] {
	return ScanIWithContext(func(ctx context.Context, accumulator R, item T, index int64) (context.Context, R) {
		return ctx, reduce(accumulator, item, index)
	}, seed)
}

// ScanIWithContext applies an accumulator function over an Observable and emits each intermediate result.
func ScanIWithContext[T, R any](reduce func(ctx context.Context, accumulator R, item T, index int64) (context.Context, R), seed R) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
			accumulator := seed
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ctx, accumulator = reduce(ctx, accumulator, value, i)
						i++

						destination.NextWithContext(ctx, accumulator)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// GroupBy groups the items emitted by an Observable according to a specified criterion,
// and emits these grouped items as Observables.
func GroupBy[T any, K comparable](iteratee func(item T) K) func(Observable[T]) Observable[Observable[T]] {
	return GroupByIWithContext(func(ctx context.Context, item T, _ int64) (context.Context, K) {
		return ctx, iteratee(item)
	})
}

// GroupByWithContext groups the items emitted by an Observable according to a specified criterion,
// and emits these grouped items as Observables.
func GroupByWithContext[T any, K comparable](iteratee func(ctx context.Context, item T) (context.Context, K)) func(Observable[T]) Observable[Observable[T]] {
	return GroupByIWithContext(func(ctx context.Context, item T, _ int64) (context.Context, K) {
		return iteratee(ctx, item)
	})
}

// GroupByI groups the items emitted by an Observable according to a specified criterion,
// and emits these grouped items as Observables.
func GroupByI[T any, K comparable](iteratee func(item T, index int64) K) func(Observable[T]) Observable[Observable[T]] {
	return GroupByIWithContext(func(ctx context.Context, item T, index int64) (context.Context, K) {
		return ctx, iteratee(item, index)
	})
}

// GroupByIWithContext groups the items emitted by an Observable according to a specified criterion,
// and emits these grouped items as Observables.
func GroupByIWithContext[T any, K comparable](iteratee func(ctx context.Context, item T, index int64) (context.Context, K)) func(Observable[T]) Observable[Observable[T]] {
	return func(source Observable[T]) Observable[Observable[T]] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[Observable[T]]) Teardown {
			groups := sync.Map{}
			i := int64(0)

			notifyAll := func(cb func(Observer[T])) {
				groups.Range(func(key, value any) bool {
					cb(value.(Observer[T])) //nolint:errcheck,forcetypeassert
					return true
				})
			}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ctx, key := iteratee(ctx, value, i)
						i++

						g, ok := groups.Load(key)
						if ok {
							g.(Observer[T]).NextWithContext(ctx, value) //nolint:errcheck,forcetypeassert
						} else if !ok {
							subject := NewUnicastSubject[T](UnicastSubjectUnlimitedBufferSize)
							groups.Store(key, subject)
							subject.NextWithContext(ctx, value)
							destination.NextWithContext(ctx, subject)
						}
					},
					func(ctx context.Context, err error) {
						destination.ErrorWithContext(ctx, err)
						notifyAll(func(o Observer[T]) { o.ErrorWithContext(ctx, err) })

						groups = sync.Map{}
					},
					func(ctx context.Context) {
						destination.CompleteWithContext(ctx)
						notifyAll(func(o Observer[T]) { o.CompleteWithContext(ctx) })

						groups = sync.Map{}
					},
				),
			)

			return func() {
				sub.Unsubscribe()
				notifyAll(func(o Observer[T]) { o.CompleteWithContext(context.TODO()) })

				groups = sync.Map{}
			}
		})
	}
}

// BufferWhen buffers the items emitted by an Observable until a second Observable emits an item.
// Then it emits the buffer and starts a new buffer. It repeats this process until the source Observable completes.
// If the boundary Observable completes, the buffer is emitted and the source Observable completes.
// If the source Observable errors, the buffer is emitted and the error is propagated.
func BufferWhen[T, B any](boundary Observable[B]) func(Observable[T]) Observable[[]T] {
	return func(source Observable[T]) Observable[[]T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			buffer := []T{}
			mu := xsync.NewMutexWithSpinlock()

			flush := func(ctx context.Context) {
				// send even if buffer is empty
				mu.Lock()

				tmp := buffer
				buffer = []T{}

				mu.Unlock()

				destination.NextWithContext(ctx, tmp)
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							mu.Lock()

							buffer = append(buffer, value)

							mu.Unlock()
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							flush(ctx)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				boundary.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value B) {
							flush(ctx)
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							flush(ctx)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			return func() {
				subscriptions.Unsubscribe()
				mu.Lock()

				buffer = []T{}

				mu.Unlock()
			}
		})
	}
}

// BufferWithTimeOrCount buffers the items emitted by an Observable for a specified time or count.
// It emits the buffer and starts a new buffer. It repeats this process until the source Observable completes.
// If the source Observable errors, the buffer is emitted and the error is propagated. If the source Observable completes,
// the buffer is emitted and the complete notification is propagated. If the specified time or count is reached,
// the buffer is emitted and a new buffer is started.
func BufferWithTimeOrCount[T any](size int, duration time.Duration) func(Observable[T]) Observable[[]T] {
	if size < 1 {
		panic(ErrBufferWithTimeOrCountWrongSize)
	}

	if duration <= 0 {
		panic(ErrBufferWithTimeOrCountWrongDuration)
	}

	return func(source Observable[T]) Observable[[]T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			buffer := []T{}
			mu := xsync.NewMutexWithSpinlock()

			flush := func(ctx context.Context) {
				// send even if buffer is empty
				mu.Lock()

				tmp := buffer
				buffer = []T{}

				mu.Unlock()

				destination.NextWithContext(ctx, tmp)
			}

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							mu.Lock()

							buffer = append(buffer, value)
							isFull := len(buffer) >= size

							mu.Unlock()

							if isFull {
								flush(ctx)
							}
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							flush(ctx)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				Interval(duration).SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value int64) {
							flush(ctx)
						},
						destination.ErrorWithContext,
						func(ctx context.Context) {
							flush(ctx)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			return func() {
				subscriptions.Unsubscribe()
				mu.Lock()

				buffer = []T{}

				mu.Unlock()
			}
		})
	}
}

// BufferWithCount buffers the items emitted by an Observable until the buffer is full.
// Then it emits the buffer and starts a new buffer. It repeats this process until the
// source Observable completes. If the source Observable errors, the buffer is emitted
// and the error is propagated. If the source Observable completes, the buffer is emitted
// and the complete notification is propagated. If the specified count is reached, the buffer
// is emitted and a new buffer is started.
// Play: https://go.dev/play/p/IXhDtSybE4R
func BufferWithCount[T any](size int) func(Observable[T]) Observable[[]T] {
	if size < 1 {
		panic(ErrBufferWithCountWrongSize)
	}

	return func(source Observable[T]) Observable[[]T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			buffer := make([]T, 0, size)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						buffer = append(buffer, value)
						if len(buffer) >= size {
							destination.NextWithContext(ctx, buffer)
							buffer = make([]T, 0, size)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if len(buffer) > 0 {
							destination.NextWithContext(ctx, buffer)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return func() {
				sub.Unsubscribe()

				buffer = []T{}
			}
		})
	}
}

// BufferWithTime buffers the items emitted by an Observable for a specified time.
// It emits the buffer and starts a new buffer. It repeats this process until the source
// Observable completes. If the source Observable errors, the buffer is emitted and the error
// is propagated. If the source Observable completes, the buffer is emitted and the complete
// notification is propagated. If the specified time is reached, the buffer is emitted and a new buffer is started.
func BufferWithTime[T any](duration time.Duration) func(Observable[T]) Observable[[]T] {
	if duration <= 0 {
		panic(ErrBufferWithTimeWrongDuration)
	}

	return BufferWhen[T](Interval(duration))
}

// WindowWhen emits an Observable that represents a window of items emitted by the source Observable.
// The window emits items when the specified boundary Observable emits an item. The window closes
// and a new window opens when the boundary Observable emits an item. If the source Observable completes,
// the window emits the complete notification and the complete notification is propagated. If the boundary
// Observable completes, the window emits the complete notification and the complete notification is propagated.
func WindowWhen[T, B any](boundary Observable[B]) func(Observable[T]) Observable[Observable[T]] {
	return func(source Observable[T]) Observable[Observable[T]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[Observable[T]]) Teardown {
			var window Subject[T]

			mu := xsync.MutexWithSpinlock{}

			flush := func(ctx context.Context, skipNew bool) {
				// reset Observable even if no notification were sent
				mu.Lock()

				tmp := window

				var newSubject Subject[T]
				if !skipNew {
					newSubject = NewUnicastSubject[T](UnicastSubjectUnlimitedBufferSize)
					window = newSubject
				}

				mu.Unlock()

				if tmp != nil { // nil on first call of flush()
					tmp.CompleteWithContext(ctx)
				}

				if !skipNew {
					destination.NextWithContext(ctx, newSubject)
				}
			}

			flush(subscriberCtx, false) // create and send first window

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							mu.Lock()

							tmp := window

							mu.Unlock()

							tmp.NextWithContext(ctx, value)
						},
						func(ctx context.Context, err error) {
							flush(ctx, true)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							flush(ctx, true)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			subscriptions.AddUnsubscribable(
				boundary.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value B) {
							flush(ctx, false)
						},
						func(ctx context.Context, err error) {
							flush(ctx, true)
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							flush(ctx, true)
							destination.CompleteWithContext(ctx)
						},
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// SampleWhen emits the most recently emitted value from the source Observable
// within a period determined by another Observable?
//
// Note that if the source Observable has emitted no items since the last
// time it was sampled, the Observable that results from this operator will
// emit no item for that sampling period.
func SampleWhen[T, t any](tick Observable[t]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var last lo.Tuple2[context.Context, T]

			var hasValue bool

			mu := xsync.NewMutexWithSpinlock()

			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							mu.Lock()

							last = lo.T2(ctx, value)
							hasValue = true

							mu.Unlock()
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			subscriptions.AddUnsubscribable(
				tick.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value t) {
							mu.Lock()

							if hasValue {
								hasValue = false
								cOpy := last

								// will be executed after mutex unlock
								defer destination.NextWithContext(cOpy.A, cOpy.B)
							}

							mu.Unlock()
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// SampleTime emits the most recently emitted value from the source Observable
// within periodic time intervals.
//
// Note that if the source Observable has emitted no items since the last
// time it was sampled, the Observable that results from this operator will
// emit no item for that sampling period.
func SampleTime[T any](interval time.Duration) func(Observable[T]) Observable[T] {
	return SampleWhen[T](
		Interval(interval),
	)
}

// ThrottleWhen emits a value from the source Observable, then ignores subsequent source
// values for a duration determined by another Observable, then repeats this process.
func ThrottleWhen[T, t any](tick Observable[t]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			// 0: don't send
			// 1: send
			var send int32

			atomic.StoreInt32(&send, 0)

			subscription := NewSubscription(nil)

			// We must subscribe to `tick` first: if a synchronous Next notification
			// is sent, the first value of `source` will be forward.
			subscription.AddUnsubscribable(
				tick.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value t) {
							atomic.StoreInt32(&send, 1)
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			subscription.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							if atomic.CompareAndSwapInt32(&send, 1, 0) {
								destination.NextWithContext(ctx, value)
							}
						},
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					),
				),
			)

			return subscription.Unsubscribe
		})
	}
}

// ThrottleTime emits a value from the source Observable, then ignores subsequent source
// values for duration milliseconds, then repeats this process.
func ThrottleTime[T any](interval time.Duration) func(Observable[T]) Observable[T] {
	intervalNano := interval.Nanoseconds()

	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			lastAt := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						now := xtime.NowNanoMonotonic()
						if lastAt+intervalNano < now {
							lastAt = now

							destination.NextWithContext(ctx, value)
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
