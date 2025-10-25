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


package rohot

import (
	"context"
	"errors"

	"github.com/samber/hot"
	"github.com/samber/lo"
	"github.com/samber/ro"
)

var NotFound = errors.New("rohot.GetOrFetchOrError: not found")

// GetOrFetch creates an operator that retrieves values from cache or fetches them when missing.
// Play: https://go.dev/play/p/7mKj3n8fH4b
func GetOrFetch[K comparable, V any](cache *hot.HotCache[K, V]) func(ro.Observable[K]) ro.Observable[lo.Tuple2[V, bool]] {
	return func(source ro.Observable[K]) ro.Observable[lo.Tuple2[V, bool]] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[lo.Tuple2[V, bool]]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value K) {
						v, ok, err := cache.Get(value)
						if err != nil {
							destination.ErrorWithContext(ctx, err)
							return
						}

						destination.NextWithContext(ctx, lo.T2(v, ok))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// GetOrFetchOrSkip creates an operator that retrieves cached values and skips missing ones.
// Play: https://go.dev/play/p/2gN9k8fJ3bL
func GetOrFetchOrSkip[K comparable, V any](cache *hot.HotCache[K, V]) func(ro.Observable[K]) ro.Observable[V] {
	return func(source ro.Observable[K]) ro.Observable[V] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[V]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value K) {
						v, ok, err := cache.Get(value)
						if err != nil {
							destination.ErrorWithContext(ctx, err)
							return
						}

						if ok {
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

// GetOrFetchOrError creates an operator that retrieves cached values or emits an error for missing ones.
// Play: https://go.dev/play/p/9hJ7k6fL2mN
func GetOrFetchOrError[K comparable, V any](cache *hot.HotCache[K, V]) func(ro.Observable[K]) ro.Observable[V] {
	return func(source ro.Observable[K]) ro.Observable[V] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[V]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value K) {
						v, ok, err := cache.Get(value)
						if err != nil {
							destination.ErrorWithContext(ctx, err)
							return
						}

						if ok {
							destination.NextWithContext(ctx, v)
						} else {
							destination.ErrorWithContext(ctx, NotFound)
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

// GetOrFetchMany creates an operator that retrieves multiple cached values at once.
// Play: https://go.dev/play/p/4hK8n7fJ3mP
func GetOrFetchMany[K comparable, V any](cache *hot.HotCache[K, V]) func(ro.Observable[[]K]) ro.Observable[map[K]V] {
	return func(source ro.Observable[[]K]) ro.Observable[map[K]V] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[map[K]V]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, values []K) {
						v, _, err := cache.GetMany(values)
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
