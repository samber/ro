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

package robackoff

import (
	"context"
	"time"

	"github.com/samber/ro"
)

type Config struct {
	MaxRetries     uint64
	Strategy       Strategy
	ResetOnSuccess bool
}

func delayFor(strategy Strategy, attempt uint64) time.Duration {
	if strategy == nil {
		return 0
	}
	return strategy(attempt)
}

func sleep(ctx context.Context, d time.Duration) bool {
	if d <= 0 {
		return ctx.Err() == nil
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

func Retry[T any](strategy Strategy) func(ro.Observable[T]) ro.Observable[T] {
	return RetryWithConfig[T](Config{Strategy: strategy})
}

func RetryWithConfig[T any](config Config) func(ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			attempt := uint64(0)

			for {
				if err := subscriberCtx.Err(); err != nil {
					destination.ErrorWithContext(subscriberCtx, err)
					return nil
				}
				if destination.IsClosed() {
					return nil
				}

				var lastErr error

				sub := source.SubscribeWithContext(
					subscriberCtx,
					ro.NewObserverWithContext(
						func(ctx context.Context, value T) {
							destination.NextWithContext(ctx, value)
						},
						func(ctx context.Context, err error) {
							lastErr = err
						},
						destination.CompleteWithContext,
					),
				)

				sub.Wait()

				if lastErr == nil {
					if config.ResetOnSuccess {
						attempt = 0
					}
					return nil
				}

				attempt++

				if config.MaxRetries > 0 && attempt > config.MaxRetries {
					destination.ErrorWithContext(subscriberCtx, lastErr)
					return nil
				}

				if !sleep(subscriberCtx, delayFor(config.Strategy, attempt)) {
					destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
					return nil
				}
			}
		})
	}
}
