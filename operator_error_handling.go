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
	"time"
)

// Catch catches errors on the observable to be handled by returning a new observable
// or throwing an error.
// Play: https://go.dev/play/p/0pVlxwjhdMT
func Catch[T any](finally func(err error) Observable[T]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			subscriptions := NewSubscription(nil)

			subscriptions.AddUnsubscribable(
				source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						destination.NextWithContext,
						func(ctx context.Context, err error) {
							subscriptions.AddUnsubscribable(
								finally(err).SubscribeWithContext(ctx, destination),
							)
						},
						destination.CompleteWithContext,
					),
				),
			)

			return subscriptions.Unsubscribe
		})
	}
}

// OnErrorResumeNextWith instructs an Observable to begin emitting a second
// Observable sequence if it encounters an error or completes. It immediately
// subscribes to the next one that was passed.
// Play: https://go.dev/play/p/9XLTAOginbK
func OnErrorResumeNextWith[T any](finally ...Observable[T]) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		if len(finally) == 0 {
			return source
		}

		finally = append([]Observable[T]{source}, finally...)

		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			subscriptions := NewSubscription(nil)

			var lastCtx context.Context

			var err error

			for i := range finally {
				if subscriptions.IsClosed() {
					break
				}

				err = nil

				sub := finally[i].SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						destination.NextWithContext,
						func(ctx context.Context, e error) {
							err = e
							lastCtx = ctx
						},
						func(ctx context.Context) {
							lastCtx = ctx
						},
					),
				)

				// `subscriptions` cancels `sub` when it unsubscribes
				// but `sub` cannot unsubscribe `subscriptions`
				subscriptions.AddUnsubscribable(sub)
				sub.Wait()
			}

			if err != nil {
				destination.ErrorWithContext(lastCtx, err)
			} else {
				destination.CompleteWithContext(lastCtx)
			}

			return subscriptions.Unsubscribe
		})
	}
}

// OnErrorReturn instructs an Observable to emit a particular item when it
// encounters an error. It will then complete the sequence.
// Play: https://go.dev/play/p/d_9xe1oedjU
func OnErrorReturn[T any](finally T) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					destination.NextWithContext,
					func(ctx context.Context, err error) {
						destination.NextWithContext(ctx, finally)
						destination.CompleteWithContext(ctx)
					},
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Retry resubscribes to the source observable when it encounters an error.
// It will retry infinitely. If you want to limit the number of retries, use
// RetryWithConfig.
// Play: https://go.dev/play/p/Llj9dT9Y3Z2
func Retry[T any]() func(Observable[T]) Observable[T] {
	return RetryWithConfig[T](RetryConfig{
		MaxRetries:     0,     // unlimited
		Delay:          0,     // disabled
		ResetOnSuccess: false, // disabled because it retries infinitely
	})
}

// RetryConfig is the configuration for the Retry operator.
type RetryConfig struct {
	MaxRetries     uint64
	Delay          time.Duration
	ResetOnSuccess bool
}

// RetryWithConfig resubscribes to the source observable when it encounters
// an error. If a max number of retries is set, it will retry until the max
// number of retries is reached. If a delay is set, it will wait before retrying.
// If resetOnSuccess is set, it will reset the number of retries when a value is
// emitted.
// Play: https://go.dev/play/p/GilWi5xG0lr
func RetryWithConfig[T any](opts RetryConfig) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			subscriptions := NewSubscription(nil)
			retries := uint64(0)

			for !subscriptions.IsClosed() {
				// Check for context cancellation before retrying
				select {
				case <-subscriberCtx.Done():
					destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
					return subscriptions.Unsubscribe
				default:
				}

				var shouldRetry bool
				var lastErr error

				sub := source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value T) {
							if opts.ResetOnSuccess {
								retries = 0
							}
							destination.NextWithContext(ctx, value)
						},
						func(ctx context.Context, err error) {
							lastErr = err
							retries++
							shouldRetry = opts.MaxRetries == 0 || retries <= opts.MaxRetries
						},
						func(ctx context.Context) {
							destination.CompleteWithContext(ctx)
						},
					),
				)

				subscriptions.AddUnsubscribable(sub)
				sub.Wait()

				if lastErr != nil {
					if shouldRetry {
						if opts.Delay > 0 {
							// Use context-aware sleep that can be cancelled
							select {
							case <-time.After(opts.Delay):
								// Continue to next iteration
							case <-subscriberCtx.Done():
								destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
								return subscriptions.Unsubscribe
							}
						}
						// Continue to next iteration
						continue
					}
					destination.ErrorWithContext(subscriberCtx, lastErr)
				}
				break
			}

			return subscriptions.Unsubscribe
		})
	}
}

// ThrowIfEmpty throws an error if the source observable is empty. It will
// throw the error returned by the throw function. If the source observable
// emits a value, it will complete. If the source observable emits an error,
// it will propagate the error.
// Play: https://go.dev/play/p/mLCaC7p_6p4
func ThrowIfEmpty[T any](throw func() error) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			count := uint64(0)
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						atomic.AddUint64(&count, 1)
						destination.NextWithContext(ctx, value)
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if atomic.LoadUint64(&count) == 0 {
							destination.ErrorWithContext(ctx, throw())
						} else {
							destination.CompleteWithContext(ctx)
						}
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// DoWhile repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/nEWabaItDpn
func DoWhile[T any](condition func() bool) func(Observable[T]) Observable[T] {
	return DoWhileI[T](func(_ int64) bool {
		return condition()
	})
}

// DoWhileWithContext repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
func DoWhileWithContext[T any](condition func(ctx context.Context) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return DoWhileIWithContext[T](func(ctx context.Context, _ int64) (context.Context, bool) {
		return condition(ctx)
	})
}

// DoWhileI repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/cxOA9gimkCq
func DoWhileI[T any](condition func(index int64) bool) func(Observable[T]) Observable[T] {
	return DoWhileIWithContext[T](func(ctx context.Context, index int64) (context.Context, bool) {
		return ctx, condition(index)
	})
}

// DoWhileIWithContext repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/yMoCCnnvRRH
func DoWhileIWithContext[T any](condition func(ctx context.Context, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)
			subscriptions := NewSubscription(nil)
			currentCtx := subscriberCtx
			shouldContinue := true
			var lastErr error

			for shouldContinue {
				if subscriptions.IsClosed() {
					break
				}

				var completed bool

				sub := source.SubscribeWithContext(
					currentCtx,
					NewObserverWithContext(
						destination.NextWithContext,
						func(ctx context.Context, err error) {
							lastErr = err
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							currentCtx, shouldContinue = condition(ctx, i)
							completed = true
							i++
						},
					),
				)

				subscriptions.AddUnsubscribable(sub)
				sub.Wait()

				if lastErr != nil {
					// Source emitted an error, stop the loop
					break
				}

				if completed && !shouldContinue {
					// Condition is false, stop the loop
					break
				}
			}

			if lastErr == nil {
				destination.CompleteWithContext(currentCtx)
			}

			return subscriptions.Unsubscribe
		})
	}
}

// While repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/hMj3DBVtp73
func While[T any](condition func() bool) func(Observable[T]) Observable[T] {
	return WhileIWithContext[T](func(ctx context.Context, _ int64) (context.Context, bool) {
		return ctx, condition()
	})
}

// WhileWithContext repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
func WhileWithContext[T any](condition func(ctx context.Context) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return WhileIWithContext[T](func(ctx context.Context, _ int64) (context.Context, bool) {
		return condition(ctx)
	})
}

// WhileI repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/9aAuzAspyMc
func WhileI[T any](condition func(index int64) bool) func(Observable[T]) Observable[T] {
	return WhileIWithContext[T](func(ctx context.Context, index int64) (context.Context, bool) {
		return ctx, condition(index)
	})
}

// WhileIWithContext repeats the source observable while the condition is true. It will
// complete when the condition is false. It will not emit any values if the
// source observable is empty. It will not emit any values if the source observable
// emits an error.
// Play: https://go.dev/play/p/xTpqdGSxOxw
func WhileIWithContext[T any](condition func(ctx context.Context, index int64) (context.Context, bool)) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)
			subscriptions := NewSubscription(nil)
			currentCtx := subscriberCtx
			var lastErr error

			for !subscriptions.IsClosed() {
				var nextCtx context.Context
				var shouldContinue bool
				nextCtx, shouldContinue = condition(currentCtx, i)

				if !shouldContinue {
					// Condition is false, stop the loop
					break
				}

				i++

				sub := source.SubscribeWithContext(
					nextCtx,
					NewObserverWithContext(
						destination.NextWithContext,
						func(ctx context.Context, err error) {
							lastErr = err
							destination.ErrorWithContext(ctx, err)
						},
						func(ctx context.Context) {
							// Source completed normally
						},
					),
				)

				subscriptions.AddUnsubscribable(sub)
				sub.Wait()

				if lastErr != nil {
					// Source emitted an error, stop the loop
					break
				}

				currentCtx = nextCtx
			}

			if lastErr == nil {
				destination.CompleteWithContext(currentCtx)
			}

			return subscriptions.Unsubscribe
		})
	}
}
