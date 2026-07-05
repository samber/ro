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


package roratelimit

import (
	"context"

	"github.com/samber/ro"
	"github.com/ulule/limiter/v3"
)

// NewRateLimiter rate limits observable values using ulule/limiter with custom key extraction.
// Play: https://go.dev/play/p/V4meCiGc3bx
func NewRateLimiter[T any](limiter *limiter.Limiter, keyGetter func(T) string) func(destination ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext( // @TODO: use unsafe observer implem, for performance ?
					func(ctx context.Context, value T) {
						key := keyGetter(value)

						rate, err := limiter.Get(ctx, key)
						if err != nil {
							destination.ErrorWithContext(ctx, err)
						} else if !rate.Reached {
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
