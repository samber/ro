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
	"time"
)

// ToSlice collects all items from the observable into a slice. It is a sink
// operator so it emit a single value. It emits the slice when the source
// completes. If the source is empty, it emits an empty slice.
// Play: https://go.dev/play/p/kxbU_PzpN6t
func ToSlice[T any]() func(Observable[T]) Observable[[]T] {
	return func(source Observable[T]) Observable[[]T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[[]T]) Teardown {
			slice := []T{}

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						slice = append(slice, value)
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, slice) // @TODO: use the context.Context from the last Next notification ?
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ToMap collects all items from the observable into a map. It is a sink
// operator so it emit a single value. It emits the map when the source
// completes. If the source is empty, it emits an empty map.
// Play: https://go.dev/play/p/FiF83XYB0ba
func ToMap[T any, K comparable, V any](project func(item T) (K, V)) func(Observable[T]) Observable[map[K]V] {
	return ToMapIWithContext(func(ctx context.Context, item T, _ int64) (K, V) {
		return project(item)
	})
}

// ToMapWithContext collects all items from the observable into a map. It is a sink
// operator so it emit a single value. It emits the map when the source
// completes. If the source is empty, it emits an empty map.
// Play: https://go.dev/play/p/FiF83XYB0ba
func ToMapWithContext[T any, K comparable, V any](project func(ctx context.Context, item T) (K, V)) func(Observable[T]) Observable[map[K]V] {
	return ToMapIWithContext(func(ctx context.Context, item T, _ int64) (K, V) {
		return project(ctx, item)
	})
}

// ToMapI collects all items from the observable into a map. It is a sink
// operator so it emit a single value. It emits the map when the source
// completes. If the source is empty, it emits an empty map.
// Play: https://go.dev/play/p/FiF83XYB0ba
func ToMapI[T any, K comparable, V any](mapper func(item T, index int64) (K, V)) func(Observable[T]) Observable[map[K]V] {
	return ToMapIWithContext(func(ctx context.Context, item T, index int64) (K, V) {
		return mapper(item, index)
	})
}

// ToMapIWithContext collects all items from the observable into a map. It is a sink
// operator so it emit a single value. It emits the map when the source
// completes. If the source is empty, it emits an empty map.
// Play: https://go.dev/play/p/FiF83XYB0ba
func ToMapIWithContext[T any, K comparable, V any](mapper func(ctx context.Context, item T, index int64) (K, V)) func(Observable[T]) Observable[map[K]V] {
	return func(source Observable[T]) Observable[map[K]V] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[map[K]V]) Teardown {
			output := map[K]V{}
			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						k, v := mapper(ctx, value, i)
						i++
						output[k] = v
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, output)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ToChannel materializes and forward all items from the observable into a
// channel. It is a sink operator so it emit a single value. It emits the
// channel when the source completes. If the source is empty, it emits an
// empty channel. The channel will be closed when the source completes or
// emit an error.
// Play: https://go.dev/play/p/WMKa26sirV0
func ToChannel[T any](size int) func(Observable[T]) Observable[<-chan Notification[T]] {
	if size < 0 {
		panic(ErrToChannelWrongSize)
	}

	return func(source Observable[T]) Observable[<-chan Notification[T]] {
		return NewObservableWithContext(func(subscriberCtx context.Context, destination Observer[<-chan Notification[T]]) Teardown {
			ch := make(chan Notification[T], size)

			once := sync.Once{}
			closeChan := func() {
				once.Do(func() {
					close(ch)
				})
			}

			subscriptions := NewSubscription(nil)

			// Send the channel to the observer, because
			// it's going to detach the upstream from the downstream.
			// The next operator might be long-running.
			go func() {
				// This is a workaround to avoid a race condition between the
				// destination.NextWithContext() and the destination.CompleteWithContext()
				// on empty source.
				time.Sleep(1 * time.Millisecond)

				subscriptions.AddUnsubscribable(
					source.SubscribeWithContext(
						subscriberCtx,
						NewObserverWithContext(
							func(ctx context.Context, value T) {
								ch <- NewNotificationNext(value)
							},
							func(ctx context.Context, err error) {
								ch <- NewNotificationError[T](err)

								closeChan()
								destination.CompleteWithContext(ctx)
							},
							func(ctx context.Context) {
								ch <- NewNotificationComplete[T]()

								closeChan()
								destination.CompleteWithContext(ctx)
							},
						),
					),
				)
			}()

			// Send the channel to the observer, after the goroutine is started.
			// Because the observer might call be long-running.
			// But on empty source, the destination.CompleteWithContext() might be
			// called before the goroutine is started.
			destination.NextWithContext(context.TODO(), ch)

			return func() {
				subscriptions.Unsubscribe()
				closeChan()
			}
		})
	}
}
