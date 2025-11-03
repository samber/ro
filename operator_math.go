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
	"math/big"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/constraints"
)

// maxPow10Chunk is the largest decimal exponent n for which 10^n fits in a
// float64 (IEEE-754). math.Pow10(308) == 1e308 is finite; math.Pow10(309)
// overflows to +Inf. The code uses math.Pow10(step) and then converts that
// finite float64 into a big.Float when constructing chunk factors. Keeping
// the step ≤ 308 prevents creating +Inf/NaN from math.Pow10 before moving to
// big.Float arithmetic.
const maxPow10Chunk = 308

// maxPow10ChunkCount caps the number of 308-digit chunks we are willing to
// process when emulating arbitrary-precision ceil operations. 32 chunks
// (32 * 308 ≈ 9856 decimal digits) keep allocations bounded while still
// covering far more precision than realistic callers require. If the required
// chunk count exceeds this value the implementation falls back to a safe
// no-op or infinite-precision handler to avoid runaway allocations.
const maxPow10ChunkCount = 32

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

// CeilWithPrecision emits the ceiling of the values emitted by the source Observable.
// It uses the provided decimal precision. Positive precisions apply the ceiling to the
// specified number of digits to the right of the decimal point, while negative
// precisions round to powers of ten.
func CeilWithPrecision(places int) func(Observable[float64]) Observable[float64] {
	if places < 0 {
		if places == math.MinInt {
			return ceilWithInfiniteNegativePrecision()
		}

		negPlaces := -places
		if negPlaces < 0 {
			return ceilWithInfiniteNegativePrecision()
		}

		if negPlaces > maxPow10Chunk {
			return ceilWithLargeNegativePrecision(negPlaces)
		}
	}

	if places > maxPow10Chunk {
		return ceilWithLargePositivePrecision(places)
	}

	factor := math.Pow10(places)

	if factor == 0 {
		return Ceil()
	}

	if places > 0 && math.IsInf(factor, 0) {
		return ceilWithLargePositivePrecision(places)
	}

	inverseFactor := 1 / factor
	if math.IsInf(inverseFactor, 0) {
		if places < 0 {
			negPlaces := -places
			if negPlaces < 0 {
				return ceilWithInfiniteNegativePrecision()
			}

			return ceilWithLargeNegativePrecision(negPlaces)
		}

		return Ceil()
	}

	var ceilWithBigFactor func(float64) float64
	var ceilWithSmallFactor func(float64) float64

	if places > 0 {
		ceilWithBigFactor = makeCeilWithBigFactor(factor)
	} else if places < 0 {
		ceilWithSmallFactor = makeCeilWithSmallFactor(factor)
	}

	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						scaled := value * factor
						if math.IsInf(scaled, 0) {
							if ceilWithBigFactor != nil {
								destination.NextWithContext(ctx, ceilWithBigFactor(value))
							} else {
								destination.NextWithContext(ctx, math.Ceil(value))
							}
							return
						}

						if places < 0 && scaled == 0 && value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0) {
							if ceilWithSmallFactor != nil {
								destination.NextWithContext(ctx, ceilWithSmallFactor(value))
							} else {
								destination.NextWithContext(ctx, math.Ceil(value))
							}
							return
						}

						ceiled := math.Ceil(scaled)
						result := ceiled * inverseFactor
						if math.IsInf(result, 0) || math.IsNaN(result) {
							if places < 0 && !math.IsNaN(value) && !math.IsInf(value, 0) && value > 0 {
								if ceilWithSmallFactor != nil {
									destination.NextWithContext(ctx, ceilWithSmallFactor(value))
								} else {
									destination.NextWithContext(ctx, math.Inf(1))
								}
							} else if ceilWithBigFactor != nil {
								destination.NextWithContext(ctx, ceilWithBigFactor(value))
							} else {
								destination.NextWithContext(ctx, math.Ceil(value))
							}
							return
						}

						destination.NextWithContext(ctx, result)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

func ceilWithInfiniteNegativePrecision() func(Observable[float64]) Observable[float64] {
	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						if math.IsNaN(value) || math.IsInf(value, 0) {
							destination.NextWithContext(ctx, math.Ceil(value))
							return
						}

						if value > 0 {
							destination.NextWithContext(ctx, math.Inf(1))
							return
						}

						destination.NextWithContext(ctx, 0)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

func ceilWithLargePositivePrecision(places int) func(Observable[float64]) Observable[float64] {
	if places >= math.MaxInt-(maxPow10Chunk-1) {
		return func(source Observable[float64]) Observable[float64] {
			return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
				sub := source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value float64) {
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

	chunkCount := (places + maxPow10Chunk - 1) / maxPow10Chunk
	if chunkCount > maxPow10ChunkCount {
		return func(source Observable[float64]) Observable[float64] {
			return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
				sub := source.SubscribeWithContext(
					subscriberCtx,
					NewObserverWithContext(
						func(ctx context.Context, value float64) {
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
	chunkFactors := make([]*big.Float, 0, chunkCount)

	for remaining := places; remaining > 0; {
		step := remaining
		if step > maxPow10Chunk {
			step = maxPow10Chunk
		}

		factor := math.Pow10(step)
		chunkFactors = append(chunkFactors, new(big.Float).SetPrec(256).SetFloat64(factor))
		remaining -= step
	}

	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						if math.IsNaN(value) || math.IsInf(value, 0) {
							destination.NextWithContext(ctx, math.Ceil(value))
							return
						}

						scaled := new(big.Float).SetPrec(256).SetFloat64(value)
						for _, factor := range chunkFactors {
							scaled.Mul(scaled, factor)
						}

						ceiled := ceilBigFloat(scaled)

						for i := len(chunkFactors) - 1; i >= 0; i-- {
							ceiled.Quo(ceiled, chunkFactors[i])
						}

						result, _ := ceiled.Float64()
						if math.IsInf(result, 0) || math.IsNaN(result) {
							destination.NextWithContext(ctx, math.Ceil(value))
							return
						}

						destination.NextWithContext(ctx, result)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

func ceilWithLargeNegativePrecision(places int) func(Observable[float64]) Observable[float64] {
	if places >= math.MaxInt-(maxPow10Chunk-1) {
		return ceilWithInfiniteNegativePrecision()
	}

	chunkCount := (places + maxPow10Chunk - 1) / maxPow10Chunk
	if chunkCount > maxPow10ChunkCount {
		return ceilWithInfiniteNegativePrecision()
	}
	chunkFactors := make([]*big.Float, 0, chunkCount)

	for remaining := places; remaining > 0; {
		step := remaining
		if step > maxPow10Chunk {
			step = maxPow10Chunk
		}

		factor := math.Pow10(step)
		chunkFactors = append(chunkFactors, new(big.Float).SetPrec(256).SetFloat64(factor))
		remaining -= step
	}

	return func(source Observable[float64]) Observable[float64] {
		return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[float64]) Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				NewObserverWithContext(
					func(ctx context.Context, value float64) {
						if math.IsNaN(value) || math.IsInf(value, 0) {
							destination.NextWithContext(ctx, math.Ceil(value))
							return
						}

						scaled := new(big.Float).SetPrec(256).SetFloat64(value)
						for _, factor := range chunkFactors {
							scaled.Quo(scaled, factor)
						}

						ceiled := ceilBigFloat(scaled)

						for i := len(chunkFactors) - 1; i >= 0; i-- {
							ceiled.Mul(ceiled, chunkFactors[i])
						}

						result, _ := ceiled.Float64()
						destination.NextWithContext(ctx, result)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

func ceilBigFloat(x *big.Float) *big.Float {
	prec := x.Prec()

	integer := new(big.Int)
	x.Int(integer)

	result := new(big.Float).SetPrec(prec).SetInt(integer)

	if x.Sign() > 0 {
		fractional := new(big.Float).SetPrec(prec)
		fractional.Sub(x, result)
		if fractional.Sign() > 0 {
			integer.Add(integer, big.NewInt(1))
			result.SetInt(integer)
		}
	}

	return result
}

// helper: create a ceiler that uses a big.Float factor (used for positive places)
func makeCeilWithBigFactor(factor float64) func(float64) float64 {
	bigFactor := new(big.Float).SetPrec(256).SetFloat64(factor)
	return func(value float64) float64 {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return math.Ceil(value)
		}

		scaled := new(big.Float).SetPrec(256).SetFloat64(value)
		scaled.Mul(scaled, bigFactor)

		ceiled := ceilBigFloat(scaled)
		ceiled.Quo(ceiled, bigFactor)

		result, _ := ceiled.Float64()
		if math.IsInf(result, 0) || math.IsNaN(result) {
			return math.Ceil(value)
		}

		return result
	}
}

// helper: create a ceiler that uses a big.Float factor (used for negative places)
func makeCeilWithSmallFactor(factor float64) func(float64) float64 {
	smallFactor := new(big.Float).SetPrec(256).SetFloat64(factor)
	return func(value float64) float64 {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return math.Ceil(value)
		}

		scaled := new(big.Float).SetPrec(256).SetFloat64(value)
		scaled.Mul(scaled, smallFactor)

		ceiled := ceilBigFloat(scaled)
		ceiled.Quo(ceiled, smallFactor)

		result, _ := ceiled.Float64()
		if math.IsInf(result, 0) || math.IsNaN(result) {
			if value > 0 {
				return math.Inf(1)
			}

			return math.Ceil(value)
		}

		return result
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
