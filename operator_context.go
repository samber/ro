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
	"time"
)

// ContextWithValue returns an Observable that emits the same items as the source
// Observable, but adds a key-value pair to the context of each item.
// Play: https://go.dev/play/p/l70D6fuiVhK
func ContextWithValue[T any](k, v any) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				context.WithValue(subscriberCtx, k, v),
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						ctx = context.WithValue(ctx, k, v)
						destination.NextWithContext(ctx, value)
					},
					func(ctx context.Context, err error) {
						ctx = context.WithValue(ctx, k, v)
						destination.ErrorWithContext(ctx, err)
					},
					func(ctx context.Context) {
						ctx = context.WithValue(ctx, k, v)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ContextWithTimeout returns an Observable that emits the same items as the source
// Observable, but adds a cancel function to the context of each item.
// This operator should be chained with ThrowOnContextCancel.
// Play: https://go.dev/play/p/1qijKGsyn0D
func ContextWithTimeout[T any](timeout time.Duration) func(Observable[T]) Observable[T] {
	// return ContextWithTimeoutCause[T](timeout, nil)
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						childCtx, _ := context.WithTimeout(ctx, timeout) //nolint:govet
						destination.NextWithContext(childCtx, value)
						// We don't cancel the timeout after calling Next(), because
						// if WithTimeout is called with ObserveOn or SubscribeOn, the
						// the context might still be in use.
						// Should we uncomment the following code?
						//
						//	if childCtx.Err() != nil {
						//		destination.ErrorWithContext(ctx, childCtx.Err())
						//	}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Commented because go added support for context.WithTimeoutCause in go 1.20
// // ContextWithTimeoutCause returns an Observable that emits the same items as the source
// // Observable, but adds a cancel function to the context of each item.
// // This operator should be chained with ThrowOnContextCancel.
// func ContextWithTimeoutCause[T any](timeout time.Duration, cause error) func(Observable[T]) Observable[T] {
// 	return func(source Observable[T]) Observable[T] {
// 		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
// 			sub := source.SubscribeWithContext(
// 				subscriberCtx,
// 				NewObserverWithContext(
// 					func(ctx context.Context, value T) {
// 						childCtx, _ := context.WithTimeoutCause(ctx, timeout, cause) //nolint:govet
// 						destination.NextWithContext(childCtx, value)
// 						// We don't cancel the timeout after calling Next(), because
// 						// if WithTimeout is called with ObserveOn or SubscribeOn, the
// 						// the context might still be in use.
// 						// Should we uncomment the following code?
// 						//
// 						//	if childCtx.Err() != nil {
// 						//		destination.ErrorWithContext(ctx, childCtx.Err())
// 						//	}
// 					},
// 					destination.ErrorWithContext,
// 					destination.CompleteWithContext,
// 				),
// 			)

// 			return sub.Unsubscribe
// 		})
// 	}
// }

// ContextWithDeadline returns an Observable that emits the same items as the source
// Observable, but adds a deadline to the context of each item.
// This operator should be chained with ThrowOnContextCancel.
// Play: https://go.dev/play/p/NPYFzhI2YDK
func ContextWithDeadline[T any](deadline time.Time) func(Observable[T]) Observable[T] {
	// return ContextWithDeadlineCause[T](deadline, nil)
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						childCtx, _ := context.WithDeadline(ctx, deadline) //nolint:govet
						destination.NextWithContext(childCtx, value)
						// We don't cancel the timeout after calling Next(), because
						// if WithDeadline is called with ObserveOn or SubscribeOn, the
						// the context might still be in use.
						// Should we uncomment the following code?
						//
						//	if childCtx.Err() != nil {
						//		destination.ErrorWithContext(ctx, childCtx.Err())
						//	}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Commented because go added support for context.WithDeadlineCause in go 1.20
// // ContextWithDeadlineCause returns an Observable that emits the same items as the source
// // Observable, but adds a deadline to the context of each item.
// // This operator should be chained with ThrowOnContextCancel.
// func ContextWithDeadlineCause[T any](deadline time.Time, cause error) func(Observable[T]) Observable[T] {
// 	return func(source Observable[T]) Observable[T] {
// 		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
// 			sub := source.SubscribeWithContext(
// 				subscriberCtx,
// 				NewObserverWithContext(
// 					func(ctx context.Context, value T) {
// 						childCtx, _ := context.WithDeadlineCause(ctx, deadline, cause) //nolint:govet
// 						destination.NextWithContext(childCtx, value)
// 						// We don't cancel the timeout after calling Next(), because
// 						// if WithDeadline is called with ObserveOn or SubscribeOn, the
// 						// the context might still be in use.
// 						// Should we uncomment the following code?
// 						//
// 						//	if childCtx.Err() != nil {
// 						//		destination.ErrorWithContext(ctx, childCtx.Err())
// 						//	}
// 					},
// 					destination.ErrorWithContext,
// 					destination.CompleteWithContext,
// 				),
// 			)

// 			return sub.Unsubscribe
// 		})
// 	}
// }

// ContextReset returns an Observable that emits the same items as the source
// Observable, but with a new context. If the new context is nil, it uses
// context.Background().
// Play: https://go.dev/play/p/PgvV0SejJpH
func ContextReset[T any](newCtx context.Context) func(Observable[T]) Observable[T] {
	if newCtx == nil {
		newCtx = context.Background()
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(_ context.Context, value T) {
						destination.NextWithContext(newCtx, value)
					},
					func(_ context.Context, err error) {
						destination.ErrorWithContext(newCtx, err)
					},
					func(_ context.Context) {
						destination.CompleteWithContext(newCtx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ContextMap returns an Observable that emits the same items as the source
// Observable, but with a new context. The project function is called for each
// item emitted by the source Observable, and the context is replaced with the
// context returned by the project function.
// Play: https://go.dev/play/p/jbshjD3sb6M
func ContextMap[T any](project func(ctx context.Context) context.Context) func(Observable[T]) Observable[T] {
	return ContextMapI[T](func(ctx context.Context, _ int64) context.Context {
		return project(ctx)
	})
}

// ContextMapI returns an Observable that emits the same items as the source
// Observable, but with a new context. The project function is called for each
// item emitted by the source Observable, and the context is replaced with the
// context returned by the project function.
// Play: https://go.dev/play/p/jbshjD3sb6M
func ContextMapI[T any](project func(ctx context.Context, index int64) context.Context) func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						destination.NextWithContext(project(ctx, i), value)

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

// ThrowOnContextCancel returns an Observable that emits the same items as the source
// Observable, but throws an error if the context is canceled. This operator should
// be chained after an operator such as ContextWithTimeout or ContextWithDeadline.
// Play: https://go.dev/play/p/K9oGdZFa-b1
func ThrowOnContextCancel[T any]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			if subscriberCtx.Err() != nil {
				destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
				return nil
			}

			done := make(chan struct{})

			go func() {
				select {
				case <-subscriberCtx.Done():
					destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
				case <-done:
					destination.CompleteWithContext(subscriberCtx)
				}
			}()

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if ctx.Err() != nil {
							destination.ErrorWithContext(ctx, ctx.Err())
							return
						}

						destination.NextWithContext(ctx, value)

						if ctx.Err() != nil {
							destination.ErrorWithContext(ctx, ctx.Err())
							return
						}
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return func() {
				sub.Unsubscribe()
				close(done)
			}
		})
	}
}
