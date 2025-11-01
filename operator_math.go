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
	"math"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/constraints"
)

// Average calculates the average of the values emitted by the source Observable.
// It emits the average when the source completes. If the source is empty, it emits NaN.
// Play: https://go.dev/play/p/B0IhFEsQAin
func Average[T constraints.Numeric]() func(Observable[T]) Observable[float64] {
	return func(source Observable[T]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sum := float64(0)
			count := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						sum += float64(value)
						count++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if count == 0 {
							destination.NextWithContext(ctx, math.NaN())
							destination.CompleteWithContext(ctx)
						}

						avg := sum / float64(count)
						destination.NextWithContext(ctx, avg)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Count counts the number of values emitted by the source Observable.
// It emits the count when the source completes.
// Play: https://go.dev/play/p/igtOxOLeHPp
func Count[T any]() func(Observable[T]) Observable[int64] {
	return func(source Observable[T]) Observable[int64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[int64]) Teardown {
			count := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						count++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, count)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Sum calculates the sum of the values emitted by the source Observable.
// It emits the sum when the source completes.
// Play: https://go.dev/play/p/b3rRlI80igo
func Sum[T constraints.Numeric]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var sum T

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						sum += value
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(ctx, sum)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Round emits the rounded values emitted by the source Observable.
// Play: https://go.dev/play/p/aXwxpsJq_BQ
func Round() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Round(value))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Min emits the minimum value emitted by the source Observable.
// It emits the minimum value when the source completes. If the source is empty,
// it emits no value.
// Play: https://go.dev/play/p/SPK3L-NvZ98
func Min[T constraints.Numeric]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var mIn lo.Tuple2[context.Context, T]

			first := true

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if first || value < mIn.B {
							mIn = lo.T2(ctx, value)
							first = false
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !first {
							destination.NextWithContext(mIn.A, mIn.B)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Max emits the maximum value emitted by the source Observable. It emits the
// maximum value when the source completes. If the source is empty, it emits no value.
// Play: https://go.dev/play/p/wWljVN6i1Ip
func Max[T constraints.Numeric]() func(Observable[T]) Observable[T] {
	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			var mAx lo.Tuple2[context.Context, T]

			first := true

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						if first || value > mAx.B {
							mAx = lo.T2(ctx, value)
							first = false
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						destination.NextWithContext(mAx.A, mAx.B)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Clamp emits the number within the inclusive lower and upper bounds.
// Play: https://go.dev/play/p/fu8O-BixXPM
func Clamp[T constraints.Numeric](lower, upper T) func(Observable[T]) Observable[T] {
	if lower > upper {
		panic(ErrClampLowerLessThanUpper)
	}

	return func(source Observable[T]) Observable[T] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						switch {
						case value < lower:
							destination.NextWithContext(ctx, lower)
						case value > upper:
							destination.NextWithContext(ctx, upper)
						default:
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

// Abs emits the absolute values emitted by the source Observable.
// Play: https://go.dev/play/p/WCzxrucg7BC
func Abs() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Abs(value))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Floor emits the floor of the values emitted by the source Observable.
// Play: https://go.dev/play/p/UulGlomv9K5
func Floor() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Floor(value))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// FloorWithPrecision emits the floored values with decimal precision applied before flooring.
// Precision must be >= 0; a negative precision causes a panic.
// NaN and +/-Inf values are propagated unchanged, matching math.Floor semantics.
func FloorWithPrecision(precision int) func(Observable[float64]) Observable[float64] {
	if precision < 0 {
		panic("ro.FloorWithPrecision: precision must be >= 0")
	}

	scale := math.Pow10(precision)

	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Floor(value*scale)/scale)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Ceil emits the ceiling of the values emitted by the source Observable.
// Play: https://go.dev/play/p/BlpeIki-oMG
func Ceil() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Ceil(value))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Trunc emits the truncated values emitted by the source Observable.
// Play: https://go.dev/play/p/iYc9oGDgRZJ
func Trunc() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						destination.NextWithContext(ctx, math.Trunc(value))
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Reduce applies an accumulator function over the source Observable, and emits
// the result when the source completes. It takes a seed value as the initial
// accumulator value.
// Play: https://go.dev/play/p/GpOF9eNpA5w
func Reduce[T, R any](accumulator func(agg R, item T) R, seed R) func(Observable[T]) Observable[R] {
	return ReduceIWithContext(func(ctx context.Context, agg R, item T, _ int64) (context.Context, R) {
		return ctx, accumulator(agg, item)
	}, seed)
}

// ReduceWithContext applies an accumulator function over the source Observable, and emits
// the result when the source completes. It takes a seed value as the initial
// accumulator value.
func ReduceWithContext[T, R any](accumulator func(ctx context.Context, agg R, item T) (context.Context, R), seed R) func(Observable[T]) Observable[R] {
	return ReduceIWithContext(func(ctx context.Context, agg R, item T, _ int64) (context.Context, R) {
		return accumulator(ctx, agg, item)
	}, seed)
}

// ReduceI applies an accumulator function over the source Observable, and emits
// the result when the source completes. It takes a seed value as the initial
// accumulator value.
func ReduceI[T, R any](accumulator func(agg R, item T, index int64) R, seed R) func(Observable[T]) Observable[R] {
	return ReduceIWithContext(func(ctx context.Context, agg R, item T, index int64) (context.Context, R) {
		return ctx, accumulator(agg, item, index)
	}, seed)
}

// ReduceIWithContext applies an accumulator function over the source Observable,
// and emits the result when the source completes. It takes a seed value as the
// initial accumulator value.
// Play: https://go.dev/play/p/WALnb341F4U
func ReduceIWithContext[T, R any](accumulator func(ctx context.Context, agg R, item T, index int64) (context.Context, R), seed R) func(Observable[T]) Observable[R] {
	return func(source Observable[T]) Observable[R] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[R]) Teardown {
			output := seed

			var lastCtx context.Context

			i := int64(0)

			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value T) {
						lastCtx, output = accumulator(ctx, output, value, i)
						i++
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if i == 0 {
							destination.NextWithContext(ctx, output)
						} else {
							destination.NextWithContext(lastCtx, output)
						}

						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}
