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


package rohyperloglog

import (
	"context"

	"github.com/axiomhq/hyperloglog"
	"github.com/samber/ro"
)

func CountDistinct[T comparable](precision uint8, sparse bool, hashFunc func(input T) uint64) func(ro.Observable[T]) ro.Observable[uint64] {
	if precision < 4 || precision > 18 {
		panic(ErrInvalidPrecision)
	}

	return func(source ro.Observable[T]) ro.Observable[uint64] {
		return ro.NewObservableWithContext[uint64](func(subscriberCtx context.Context, destination ro.Observer[uint64]) ro.Teardown {
			// the error is handled above
			sketch, _ := hyperloglog.NewSketch(precision, sparse)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						sketch.InsertHash(hashFunc(value))
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, sketch.Estimate())
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

func CountDistinctReduce[T comparable](precision uint8, sparse bool, hashFunc func(input T) uint64) func(ro.Observable[T]) ro.Observable[uint64] {
	if precision < 4 || precision > 18 {
		panic(ErrInvalidPrecision)
	}

	return func(source ro.Observable[T]) ro.Observable[uint64] {
		return ro.NewObservableWithContext[uint64](func(subscriberCtx context.Context, destination ro.Observer[uint64]) ro.Teardown {
			// the error is handled above
			sketch, _ := hyperloglog.NewSketch(precision, sparse)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						sketch.InsertHash(hashFunc(value))
						destination.NextWithContext(ctx, sketch.Estimate())
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}
