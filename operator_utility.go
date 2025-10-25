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

// Tap allows you to perform side effects for notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/oDI3d6553MI
func Tap[T any](onNext func(value T), onError func(err error), onComplete func()) func(Observable[T]) Observable[T] {
	return TapWithContext(
		func(ctx context.Context, value T) {
			onNext(value)
		},
		func(ctx context.Context, err error) {
			onError(err)
		},
		func(ctx context.Context) {
			onComplete()
		},
	)
}

// TapWithContext allows you to perform side effects for notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/oDI3d6553MI
func TapWithContext[T any](onNext func(ctx context.Context, value T), onError func(ctx context.Context, err error), onComplete func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						onNext(ctx, value)
						destination.NextWithContext(ctx, value)
					},
					func(ctx context.Context, err error) {
						onError(ctx, err)
						destination.ErrorWithContext(ctx, err)
					},
					func(ctx context.Context) {
						onComplete(ctx)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Do is an alias to Tap.
// Play: https://go.dev/play/p/s_BSHgxdjUR
func Do[T any](onNext func(value T), onError func(err error), onComplete func()) func(Observable[T]) Observable[T] {
	return Tap(onNext, onError, onComplete)
}

// DoWithContext is an alias to Tap.
func DoWithContext[T any](onNext func(ctx context.Context, value T), onError func(ctx context.Context, err error), onComplete func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return TapWithContext(onNext, onError, onComplete)
}

// TapOnNext allows you to perform side effects for Next notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/oDI3d6553MI
func TapOnNext[T any](onNext func(value T)) func(Observable[T]) Observable[T] {
	return Tap(onNext, func(err error) {}, func() {})
}

// TapOnNextWithContext allows you to perform side effects for Next notifications from the source Observable
// Play: https://go.dev/play/p/oDI3d6553MI
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
func TapOnNextWithContext[T any](onNext func(ctx context.Context, value T)) func(Observable[T]) Observable[T] {
	return TapWithContext(onNext, func(ctx context.Context, err error) {}, func(ctx context.Context) {})
}

// DoOnNext is an alias to TapOnNext.
func DoOnNext[T any](onNext func(value T)) func(Observable[T]) Observable[T] {
	return TapOnNext(onNext)
}

// DoOnNextWithContext is an alias to TapOnNextWithContext.
func DoOnNextWithContext[T any](onNext func(ctx context.Context, value T)) func(Observable[T]) Observable[T] {
	return TapOnNextWithContext(onNext)
}

// TapOnError allows you to perform side effects for Error notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/oDI3d6553MI
func TapOnError[T any](onError func(err error)) func(Observable[T]) Observable[T] {
	return Tap(func(value T) {}, onError, func() {})
}

// TapOnErrorWithContext allows you to perform side effects for Error notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
func TapOnErrorWithContext[T any](onError func(ctx context.Context, err error)) func(Observable[T]) Observable[T] {
	return TapWithContext(func(ctx context.Context, value T) {}, onError, func(ctx context.Context) {})
}

// DoOnError is an alias to TapOnError.
func DoOnError[T any](onError func(err error)) func(Observable[T]) Observable[T] {
	return Tap(func(value T) {}, onError, func() {})
}

// DoOnErrorWithContext is an alias to TapOnErrorWithContext.
func DoOnErrorWithContext[T any](onError func(ctx context.Context, err error)) func(Observable[T]) Observable[T] {
	return TapWithContext(func(ctx context.Context, value T) {}, onError, func(ctx context.Context) {})
}

// TapOnComplete allows you to perform side effects for Complete notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/z1sntT6bplM
func TapOnComplete[T any](onComplete func()) func(Observable[T]) Observable[T] {
	return Tap(func(value T) {}, func(err error) {}, onComplete)
}

// TapOnCompleteWithContext allows you to perform side effects for Complete notifications from the source Observable
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/3k25j_D1OTW
func TapOnCompleteWithContext[T any](onComplete func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return TapWithContext(func(ctx context.Context, value T) {}, func(ctx context.Context, err error) {}, onComplete)
}

// DoOnComplete is an alias to TapOnComplete.
func DoOnComplete[T any](onComplete func()) func(Observable[T]) Observable[T] {
	return Tap(func(value T) {}, func(err error) {}, onComplete)
}

// DoOnCompleteWithContext is an alias to TapOnCompleteWithContext.
func DoOnCompleteWithContext[T any](onComplete func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return TapWithContext(func(ctx context.Context, value T) {}, func(ctx context.Context, err error) {}, onComplete)
}

// TapOnSubscribe allows you to perform side effects when the source Observable is subscribed to
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/0YzsxpRkO4T
func TapOnSubscribe[T any](onSubscribe func()) func(Observable[T]) Observable[T] {
	return TapOnSubscribeWithContext[T](func(ctx context.Context) {
		onSubscribe()
	})
}

// TapOnSubscribeWithContext allows you to perform side effects when the source Observable is subscribed to
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
func TapOnSubscribeWithContext[T any](onSubscribe func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			onSubscribe(subscriberCtx) // triggers before the source is subscribed
			sub := source.SubscribeWithContext(subscriberCtx, destination)

			return sub.Unsubscribe
		})
	}
}

// DoOnSubscribe is an alias to TapOnSubscribe.
func DoOnSubscribe[T any](onSubscribe func()) func(Observable[T]) Observable[T] {
	return TapOnSubscribe[T](onSubscribe)
}

// DoOnSubscribeWithContext is an alias to TapOnSubscribe.
func DoOnSubscribeWithContext[T any](onSubscribe func(ctx context.Context)) func(Observable[T]) Observable[T] {
	return TapOnSubscribeWithContext[T](onSubscribe)
}

// TapOnFinalize allows you to perform side effects when the source Observable is unsubscribed from
// without modifying the emitted items. It mirrors the source Observable and forwards its emissions
// to the provided observer.
// Play: https://go.dev/play/p/VEACE_KhdvU
func TapOnFinalize[T any](onFinalize func()) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(subscriberCtx, destination)

			return func() {
				sub.Unsubscribe()
				onFinalize() // triggers after the source is unsubscribed
			}
		})
	}
}

// DoOnFinalize is an alias to TapOnFinalize.
// Play: https://go.dev/play/p/7en6T1q33WF
func DoOnFinalize[T any](onFinalize func()) func(Observable[T]) Observable[T] {
	return TapOnFinalize[T](onFinalize)
}

// IntervalValue is a value emitted by the `TimeInterval` operator.
type IntervalValue[T any] struct {
	Value    T
	Interval time.Duration
}

// TimeInterval emits the values emitted by the source Observable with the time elapsed between each emission.
// Play: https://go.dev/play/p/VX73ZL74hPk
func TimeInterval[T any]() func(Observable[T]) Observable[IntervalValue[T]] {
	return func(source Observable[T]) Observable[IntervalValue[T]] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[IntervalValue[T]]) Teardown {
			previous := xtime.NowNanoMonotonic()

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						now := xtime.NowNanoMonotonic()
						destination.NextWithContext(ctx, IntervalValue[T]{
							Value:    value,
							Interval: time.Duration(now - previous),
						})
						previous = now
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// TimestampValue is a value emitted by the `TimeInterval` operator.
type TimestampValue[T any] struct {
	Value     T
	Timestamp time.Duration
}

// Timestamp emits the values emitted by the source Observable with the time elapsed since the source Observable was subscribed to.
// Play: https://go.dev/play/p/cDiCr6qIE2P
func Timestamp[T any]() func(Observable[T]) Observable[TimestampValue[T]] {
	return func(source Observable[T]) Observable[TimestampValue[T]] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[TimestampValue[T]]) Teardown {
			start := xtime.NowNanoMonotonic()

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						destination.NextWithContext(ctx, TimestampValue[T]{
							Value:     value,
							Timestamp: time.Duration(xtime.NowNanoMonotonic() - start),
						})
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Delay delays the emissions of the source Observable by a given duration without modifying the emitted items.
// It mirrors the source Observable and forwards its emissions to the provided observer.
// Error and Complete notifications are delayed as well.
//
// @TODO: set queue size ?
// Play: https://go.dev/play/p/K3md7WPtZGI
func Delay[T any](duration time.Duration) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			// Using time.AfterFunc is not convenient because it may introduce race
			// conditions and/or change message order.
			// We need a double mutex to prevent message reordering:
			//   - one to protect the queue and allow pushing new values while we call destination.Next()
			//   - one to protect the call to destination.Next() itself
			muQueue := xsync.NewMutexWithSpinlock()
			muNext := sync.Mutex{}
			queue := []lo.Tuple2[context.Context, Notification[T]]{}

			consume := func() {
				muQueue.Lock()

				if len(queue) == 0 {
					muQueue.Unlock()
					return
				}

				first := queue[0]
				queue = queue[1:]

				muNext.Lock()
				muQueue.Unlock()

				_ = processNotificationWithObserverAndContext(
					first.A,
					first.B,
					destination,
				)

				muNext.Unlock()
			}

			produce := func(ctx context.Context, notif Notification[T]) {
				muQueue.Lock()

				queue = append(queue, lo.T2(ctx, notif))

				muQueue.Unlock()

				time.AfterFunc(
					duration,
					consume,
				)
			}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						produce(ctx, NewNotificationNext(value))
					},
					func(ctx context.Context, err error) {
						produce(ctx, NewNotificationError[T](err))
					},
					func(ctx context.Context) {
						produce(ctx, NewNotificationComplete[T]())
					},
				),
			)

			return func() {
				sub.Unsubscribe()

				muQueue.Lock()

				queue = []lo.Tuple2[context.Context, Notification[T]]{}

				muQueue.Unlock()
			}
		})
	}
}

// DelayEach delays the emissions of the source Observable by a given duration without modifying the emitted items.
// Play: https://go.dev/play/p/dReP7-bffEU
func DelayEach[T any](duration time.Duration) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						time.Sleep(duration)
						destination.NextWithContext(ctx, value)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// RepeatWith repeats the source Observable a specified number of times.
// This is a pipeable operator. The creation operator equivalent is `Repeat`.
//
// The destination is flatten.
// Play: https://go.dev/play/p/fEKtAX9_nYe
func RepeatWith[T any](count int64) func(Observable[T]) Observable[T] {
	if count < 0 {
		panic(ErrRepeatWithWrongCount)
	}

	return func(source Observable[T]) Observable[T] {
		if count == 0 {
			return Empty[T]()
		}

		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var lastCtx context.Context

			for i := int64(0); i < count; i++ {
				source.
					SubscribeWithContext(
						subscriberCtx,
						NewObserverWithContext(
							destination.NextWithContext,
							destination.ErrorWithContext,
							func(ctx context.Context) {
								lastCtx = ctx
							},
						),
					).
					Wait()

				if destination.IsClosed() {
					break
				}
			}

			destination.CompleteWithContext(lastCtx) // might do nothing if already closed

			return nil
		})
	}
}

// Timeout raises an error if the source Observable does not emit any item within the specified duration.
// Play: https://go.dev/play/p/t0xKoj-_AqZ
func Timeout[T any](duration time.Duration) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var sub Subscription

			var lastCtx atomic.Value

			lastCtx.Store(subscriberCtx) // if no value is emitted, we use the subscriber context

			timer := time.AfterFunc(duration, func() {
				destination.ErrorWithContext(lastCtx.Load().(context.Context), newTimeoutError(duration)) //nolint:errcheck,forcetypeassert
			})

			sub = source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						timer.Stop()
						destination.NextWithContext(ctx, value)
						// @TODO: what happens if the above line is too slow?
						timer.Reset(duration)
						lastCtx.Store(ctx)
					},
					func(ctx context.Context, err error) {
						timer.Stop()
						destination.ErrorWithContext(ctx, err)
					},
					func(ctx context.Context) {
						timer.Stop()
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return func() {
				timer.Stop()
				sub.Unsubscribe()
			}
		})
	}
}

// Materialize converts the source Observable into a stream of Notification instances.
// Play: https://go.dev/play/p/ZHtPviPoqWK
func Materialize[T any]() func(Observable[T]) Observable[Notification[T]] {
	return func(source Observable[T]) Observable[Notification[T]] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[Notification[T]]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						destination.NextWithContext(ctx, NewNotificationNext(value))
					},
					func(ctx context.Context, err error) {
						destination.NextWithContext(ctx, NewNotificationError[T](err))
						destination.CompleteWithContext(ctx)
					},
					func(ctx context.Context) {
						destination.NextWithContext(ctx, NewNotificationComplete[T]())
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Dematerialize converts the source Observable of Notification instances back into a stream of items.
// Play: https://go.dev/play/p/oRymdDqkh25
func Dematerialize[T any]() func(Observable[Notification[T]]) Observable[T] {
	return func(source Observable[Notification[T]]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, notif Notification[T]) {
						processNotificationWithObserverAndContext(
							ctx,
							notif,
							destination,
						)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubscribeOn schedule the upstream flow to a different goroutine. Next, Error and Complete notifications
// are sent to a queue first, then the consumer consume this queue.
// SubscribeOn converts a push-based Observable into a pullable stream with backpressure capabilities.
//
// To schedule the downstream flow to a different goroutine, refer to SubscribeOn.
//
// When an Observable emits values faster than they can be consumed, SubscribeOn buffers these values
// in a queue of specified capacity. This allows downstream consumers to pull values at their own pace
// while managing backpressure from upstream emissions.
//
// Note: Once the buffer reaches its capacity, upstream emissions will block until space becomes
// available, effectively implementing backpressure control.
//
// @TODO: add a backpressure policy ? drop vs block.
// Play: https://go.dev/play/p/WrsTUq6yxtO
func SubscribeOn[T any](bufferSize int) func(Observable[T]) Observable[T] {
	if bufferSize <= 0 {
		panic(ErrSubscribeOnWrongBufferSize)
	}

	return detachOn[T](bufferSize, true, false)
}

// ObserveOn schedule the downstream flow to a different goroutine. Next, Error and Complete notifications
// are sent to a queue first, then the consumer consume this queue.
// ObserveOn converts a push-based Observable into a pullable stream with backpressure capabilities.
//
// To schedule the upstream flow to a different goroutine, refer to SubscribeOn.
//
// When an Observable emits values faster than they can be consumed, ObserveOn buffers these values
// in a queue of specified capacity. This allows downstream consumers to pull values at their own pace
// while managing backpressure from upstream emissions.
//
// Note: Once the buffer reaches its capacity, upstream emissions will block until space becomes
// available, effectively implementing backpressure control.
//
// @TODO: add a backpressure policy ? drop vs block.
// Play: https://go.dev/play/p/BpdKJ6Mya03
func ObserveOn[T any](bufferSize int) func(Observable[T]) Observable[T] {
	if bufferSize <= 0 {
		panic(ErrObserveOnWrongBufferSize)
	}

	return detachOn[T](bufferSize, false, true)
}

func detachOn[T any](bufferSize int, onUpstream, onDownstream bool) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			ch := make(chan lo.Tuple2[context.Context, Notification[T]], bufferSize)

			once := sync.Once{}
			stop := func() {
				once.Do(func() {
					close(ch)
				})
			}

			subscriptions := NewSubscription(nil)

			consumeUpstream := func() {
				subscriptions.AddUnsubscribable(
					source.SubscribeWithContext(
						subscriberCtx,
						NewObserverWithContext(
							func(ctx context.Context, value T) {
								ch <- lo.T2(ctx, NewNotificationNext(value))
							},
							func(ctx context.Context, err error) {
								ch <- lo.T2(ctx, NewNotificationError[T](err))

								stop()
							},
							func(ctx context.Context) {
								ch <- lo.T2(ctx, NewNotificationComplete[T]())

								stop()
							},
						),
					),
				)
			}

			produceDownstream := func() {
				for notification := range ch {
					processNotificationWithContext(
						notification.A,
						notification.B,
						destination.NextWithContext,
						destination.ErrorWithContext,
						destination.CompleteWithContext,
					)
				}
			}

			// The goroutine could be used either on producer or consumer side.
			// 	* ObserveOn moves the goroutine on the consumer side.
			// 	* SubscribeOn moves the goroutine on the producer side.

			switch {
			case onUpstream:
				go recoverUnhandledError(func() {
					consumeUpstream()
				})

				produceDownstream()
			case onDownstream:
				go recoverUnhandledError(func() {
					produceDownstream()
				})

				consumeUpstream()
			default:
				panic(ErrDetachOnWrongMode)
			}

			return func() {
				subscriptions.Unsubscribe()
				stop()
			}
		})
	}
}

// Serialize ensures thread-safe message passing by wrapping any observable in a ro.SafeObservable implementation.
func Serialize[T any]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewSafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(subscriberCtx, destination)
			return sub.Unsubscribe
		})
	}
}
