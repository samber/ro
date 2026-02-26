//go:build go1.26 && goexperiment.simd && amd64

package simd

import (
	"context"
	"math"
	"simd/archsimd"

	"github.com/samber/ro"
)

// AVX2 Int8x32 variants

// AddInt8x32 adds a scalar value to each element of int8x32 vectors.
func AddInt8x32[T ~int8](number T) func(ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
	vector := archsimd.BroadcastInt8x32(int8(number))
	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubInt8x32 subtracts a scalar value from each element of int8x32 vectors.
func SubInt8x32[T ~int8](number T) func(ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
	vector := archsimd.BroadcastInt8x32(int8(number))
	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampInt8x32 clamps each element of int8x32 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt8x32[T ~int8](minValue, maxValue T) func(ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt8x32(int8(minValue))
	maxVec := archsimd.BroadcastInt8x32(int8(maxValue))

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinInt8x32 applies a minimum bound to each element of int8x32 vectors.
func MinInt8x32[T ~int8](minValue T) func(ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
	minVec := archsimd.BroadcastInt8x32(int8(minValue))

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxInt8x32 applies a maximum bound to each element of int8x32 vectors.
func MaxInt8x32[T ~int8](maxValue T) func(ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
	maxVec := archsimd.BroadcastInt8x32(int8(maxValue))

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[*archsimd.Int8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumInt8x32 reduces int8x32 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int8 arithmetic,
// which can only hold values from -128 to 127. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumInt8x32[T ~int8]() func(ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int8
						accumulation.Store(&buf)
						total := int8(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinInt8x32 reduces int8x32 vectors to their minimum value.
func ReduceMinInt8x32[T ~int8]() func(ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int8
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxInt8x32 reduces int8x32 vectors to their maximum value.
func ReduceMaxInt8x32[T ~int8]() func(ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int8
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Float32x8 AVX2 operators (most commonly used AVX2 type)

// AddFloat32x8 adds a scalar value to each element of float32x8 vectors.
func AddFloat32x8[T ~float32](number T) func(ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
	vector := archsimd.BroadcastFloat32x8(float32(number))
	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubFloat32x8 subtracts a scalar value from each element of float32x8 vectors.
func SubFloat32x8[T ~float32](number T) func(ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
	vector := archsimd.BroadcastFloat32x8(float32(number))
	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampFloat32x8 clamps each element of float32x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampFloat32x8[T ~float32](minValue, maxValue T) func(ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastFloat32x8(float32(minValue))
	maxVec := archsimd.BroadcastFloat32x8(float32(maxValue))

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinFloat32x8 applies a minimum bound to each element of float32x8 vectors.
func MinFloat32x8[T ~float32](minValue T) func(ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
	minVec := archsimd.BroadcastFloat32x8(float32(minValue))

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxFloat32x8 applies a maximum bound to each element of float32x8 vectors.
func MaxFloat32x8[T ~float32](maxValue T) func(ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
	maxVec := archsimd.BroadcastFloat32x8(float32(maxValue))

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[*archsimd.Float32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumFloat32x8 reduces float32x8 vectors to their sum.
func ReduceSumFloat32x8[T ~float32]() func(ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float32
						accumulation.Store(&buf)
						total := float32(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinFloat32x8 reduces float32x8 vectors to their minimum value.
func ReduceMinFloat32x8[T ~float32]() func(ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							result = float32(math.Min(float64(result), float64(buf[i])))
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxFloat32x8 reduces float32x8 vectors to their maximum value.
func ReduceMaxFloat32x8[T ~float32]() func(ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							result = float32(math.Max(float64(result), float64(buf[i])))
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int32x8 AVX2 operators

// AddInt32x8 adds a scalar value to each element of int32x8 vectors.
func AddInt32x8[T ~int32](number T) func(ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
	vector := archsimd.BroadcastInt32x8(int32(number))
	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubInt32x8 subtracts a scalar value from each element of int32x8 vectors.
func SubInt32x8[T ~int32](number T) func(ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
	vector := archsimd.BroadcastInt32x8(int32(number))
	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampInt32x8 clamps each element of int32x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt32x8[T ~int32](minValue, maxValue T) func(ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt32x8(int32(minValue))
	maxVec := archsimd.BroadcastInt32x8(int32(maxValue))

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinInt32x8 applies a minimum bound to each element of int32x8 vectors.
func MinInt32x8[T ~int32](minValue T) func(ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
	minVec := archsimd.BroadcastInt32x8(int32(minValue))

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxInt32x8 applies a maximum bound to each element of int32x8 vectors.
func MaxInt32x8[T ~int32](maxValue T) func(ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
	maxVec := archsimd.BroadcastInt32x8(int32(maxValue))

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[*archsimd.Int32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumInt32x8 reduces int32x8 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int32 arithmetic.
// For extremely large sums that may exceed int32 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumInt32x8[T ~int32]() func(ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int32
						accumulation.Store(&buf)
						total := int32(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinInt32x8 reduces int32x8 vectors to their minimum value.
func ReduceMinInt32x8[T ~int32]() func(ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxInt32x8 reduces int32x8 vectors to their maximum value.
func ReduceMaxInt32x8[T ~int32]() func(ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int16x16 AVX2 operators

// AddInt16x16 adds a scalar value to each element of int16x16 vectors.
func AddInt16x16[T ~int16](number T) func(ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
	vector := archsimd.BroadcastInt16x16(int16(number))
	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubInt16x16 subtracts a scalar value from each element of int16x16 vectors.
func SubInt16x16[T ~int16](number T) func(ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
	vector := archsimd.BroadcastInt16x16(int16(number))
	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampInt16x16 clamps each element of int16x16 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt16x16[T ~int16](minValue, maxValue T) func(ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt16x16(int16(minValue))
	maxVec := archsimd.BroadcastInt16x16(int16(maxValue))

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinInt16x16 applies a minimum bound to each element of int16x16 vectors.
func MinInt16x16[T ~int16](minValue T) func(ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
	minVec := archsimd.BroadcastInt16x16(int16(minValue))

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxInt16x16 applies a maximum bound to each element of int16x16 vectors.
func MaxInt16x16[T ~int16](maxValue T) func(ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
	maxVec := archsimd.BroadcastInt16x16(int16(maxValue))

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[*archsimd.Int16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumInt16x16 reduces int16x16 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int16 arithmetic,
// which can only hold values from -32768 to 32767. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumInt16x16[T ~int16]() func(ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int16
						accumulation.Store(&buf)
						total := int16(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinInt16x16 reduces int16x16 vectors to their minimum value.
func ReduceMinInt16x16[T ~int16]() func(ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int16
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxInt16x16 reduces int16x16 vectors to their maximum value.
func ReduceMaxInt16x16[T ~int16]() func(ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int16
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint8x32 AVX2 operators

// AddUint8x32 adds a scalar value to each element of uint8x32 vectors.
func AddUint8x32[T ~uint8](number T) func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
	vector := archsimd.BroadcastUint8x32(uint8(number))
	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubUint8x32 subtracts a scalar value from each element of uint8x32 vectors.
func SubUint8x32[T ~uint8](number T) func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
	vector := archsimd.BroadcastUint8x32(uint8(number))
	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampUint8x32 clamps each element of uint8x32 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint8x32[T ~uint8](minValue, maxValue T) func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint8x32(uint8(minValue))
	maxVec := archsimd.BroadcastUint8x32(uint8(maxValue))

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinUint8x32 applies a minimum bound to each element of uint8x32 vectors.
func MinUint8x32[T ~uint8](minValue T) func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
	minVec := archsimd.BroadcastUint8x32(uint8(minValue))

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxUint8x32 applies a maximum bound to each element of uint8x32 vectors.
func MaxUint8x32[T ~uint8](maxValue T) func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
	maxVec := archsimd.BroadcastUint8x32(uint8(maxValue))

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[*archsimd.Uint8x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumUint8x32 reduces uint8x32 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint8 arithmetic,
// which can only hold values from 0 to 255. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumUint8x32[T ~uint8]() func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint8
						accumulation.Store(&buf)
						total := uint8(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinUint8x32 reduces uint8x32 vectors to their minimum value.
func ReduceMinUint8x32[T ~uint8]() func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint8
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxUint8x32 reduces uint8x32 vectors to their maximum value.
func ReduceMaxUint8x32[T ~uint8]() func(ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint8x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x32) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint8
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint16x16 AVX2 operators

// AddUint16x16 adds a scalar value to each element of uint16x16 vectors.
func AddUint16x16[T ~uint16](number T) func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
	vector := archsimd.BroadcastUint16x16(uint16(number))
	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubUint16x16 subtracts a scalar value from each element of uint16x16 vectors.
func SubUint16x16[T ~uint16](number T) func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
	vector := archsimd.BroadcastUint16x16(uint16(number))
	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampUint16x16 clamps each element of uint16x16 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint16x16[T ~uint16](minValue, maxValue T) func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint16x16(uint16(minValue))
	maxVec := archsimd.BroadcastUint16x16(uint16(maxValue))

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinUint16x16 applies a minimum bound to each element of uint16x16 vectors.
func MinUint16x16[T ~uint16](minValue T) func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
	minVec := archsimd.BroadcastUint16x16(uint16(minValue))

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxUint16x16 applies a maximum bound to each element of uint16x16 vectors.
func MaxUint16x16[T ~uint16](maxValue T) func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
	maxVec := archsimd.BroadcastUint16x16(uint16(maxValue))

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[*archsimd.Uint16x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumUint16x16 reduces uint16x16 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint16 arithmetic.
// For extremely large sums that may exceed uint16 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint16x16[T ~uint16]() func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint16
						accumulation.Store(&buf)
						total := uint16(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinUint16x16 reduces uint16x16 vectors to their minimum value.
func ReduceMinUint16x16[T ~uint16]() func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint16
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxUint16x16 reduces uint16x16 vectors to their maximum value.
func ReduceMaxUint16x16[T ~uint16]() func(ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint16x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x16) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint16
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint32x8 AVX2 operators

// AddUint32x8 adds a scalar value to each element of uint32x8 vectors.
func AddUint32x8[T ~uint32](number T) func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
	vector := archsimd.BroadcastUint32x8(uint32(number))
	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubUint32x8 subtracts a scalar value from each element of uint32x8 vectors.
func SubUint32x8[T ~uint32](number T) func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
	vector := archsimd.BroadcastUint32x8(uint32(number))
	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampUint32x8 clamps each element of uint32x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint32x8[T ~uint32](minValue, maxValue T) func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint32x8(uint32(minValue))
	maxVec := archsimd.BroadcastUint32x8(uint32(maxValue))

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinUint32x8 applies a minimum bound to each element of uint32x8 vectors.
func MinUint32x8[T ~uint32](minValue T) func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
	minVec := archsimd.BroadcastUint32x8(uint32(minValue))

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxUint32x8 applies a maximum bound to each element of uint32x8 vectors.
func MaxUint32x8[T ~uint32](maxValue T) func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
	maxVec := archsimd.BroadcastUint32x8(uint32(maxValue))

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[*archsimd.Uint32x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumUint32x8 reduces uint32x8 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint32 arithmetic.
// For extremely large sums that may exceed uint32 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint32x8[T ~uint32]() func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint32
						accumulation.Store(&buf)
						total := uint32(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinUint32x8 reduces uint32x8 vectors to their minimum value.
func ReduceMinUint32x8[T ~uint32]() func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxUint32x8 reduces uint32x8 vectors to their maximum value.
func ReduceMaxUint32x8[T ~uint32]() func(ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint32x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x8) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint32
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Float64x4 AVX2 operators

// AddFloat64x4 adds a scalar value to each element of float64x4 vectors.
func AddFloat64x4[T ~float64](number T) func(ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
	vector := archsimd.BroadcastFloat64x4(float64(number))
	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubFloat64x4 subtracts a scalar value from each element of float64x4 vectors.
func SubFloat64x4[T ~float64](number T) func(ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
	vector := archsimd.BroadcastFloat64x4(float64(number))
	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ClampFloat64x4 clamps each element of float64x4 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampFloat64x4[T ~float64](minValue, maxValue T) func(ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastFloat64x4(float64(minValue))
	maxVec := archsimd.BroadcastFloat64x4(float64(maxValue))

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						clamped := value.Max(minVec).Min(maxVec)
						destination.NextWithContext(ctx, &clamped)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MinFloat64x4 applies a minimum bound to each element of float64x4 vectors.
func MinFloat64x4[T ~float64](minValue T) func(ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
	minVec := archsimd.BroadcastFloat64x4(float64(minValue))

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						adjusted := value.Max(minVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// MaxFloat64x4 applies a maximum bound to each element of float64x4 vectors.
func MaxFloat64x4[T ~float64](maxValue T) func(ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
	maxVec := archsimd.BroadcastFloat64x4(float64(maxValue))

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[*archsimd.Float64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						adjusted := value.Min(maxVec)
						destination.NextWithContext(ctx, &adjusted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumFloat64x4 reduces float64x4 vectors to their sum.
func ReduceSumFloat64x4[T ~float64]() func(ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
	const lanes = simdLanes4

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x4
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float64
						accumulation.Store(&buf)
						total := float64(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMinFloat64x4 reduces float64x4 vectors to their minimum value.
func ReduceMinFloat64x4[T ~float64]() func(ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
	const lanes = simdLanes4

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x4
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Min(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float64
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] < result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceMaxFloat64x4 reduces float64x4 vectors to their maximum value.
func ReduceMaxFloat64x4[T ~float64]() func(ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
	const lanes = simdLanes4

	return func(source ro.Observable[*archsimd.Float64x4]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x4
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x4) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Max(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]float64
						accumulation.Store(&buf)
						result := buf[0]
						for i := uint(1); i < lanes; i++ {
							if buf[i] > result {
								result = buf[i]
							}
						}

						destination.NextWithContext(ctx, T(result))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int64x4 AVX2 operators

// AddInt64x4 adds a scalar value to each element of int64x4 vectors.
func AddInt64x4[T ~int64](number T) func(ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
	vector := archsimd.BroadcastInt64x4(int64(number))
	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubInt64x4 subtracts a scalar value from each element of int64x4 vectors.
func SubInt64x4[T ~int64](number T) func(ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
	vector := archsimd.BroadcastInt64x4(int64(number))
	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumInt64x4 reduces int64x4 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int64 arithmetic.
// For extremely large sums that may exceed int64 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumInt64x4[T ~int64]() func(ro.Observable[*archsimd.Int64x4]) ro.Observable[T] {
	const lanes = simdLanes4

	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x4
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]int64
						accumulation.Store(&buf)
						total := int64(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint64x4 AVX2 operators

// AddUint64x4 adds a scalar value to each element of uint64x4 vectors.
func AddUint64x4[T ~uint64](number T) func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
	vector := archsimd.BroadcastUint64x4(uint64(number))
	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
						added := value.Add(vector)
						destination.NextWithContext(ctx, &added)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// SubUint64x4 subtracts a scalar value from each element of uint64x4 vectors.
func SubUint64x4[T ~uint64](number T) func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
	vector := archsimd.BroadcastUint64x4(uint64(number))
	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
						subtracted := value.Sub(vector)
						destination.NextWithContext(ctx, &subtracted)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
}

// ReduceSumUint64x4 reduces uint64x4 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint64 arithmetic.
// For extremely large sums that may exceed uint64 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint64x4[T ~uint64]() func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[T] {
	const lanes = simdLanes4

	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x4
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
						if !initialized {
							accumulation = *value
							initialized = true
						} else {
							accumulation = accumulation.Add(*value)
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if !initialized {
							destination.NextWithContext(ctx, T(0))
							destination.CompleteWithContext(ctx)
							return
						}

						var buf [lanes]uint64
						accumulation.Store(&buf)
						total := uint64(0)
						for i := uint(0); i < lanes; i++ {
							total += buf[i]
						}

						destination.NextWithContext(ctx, T(total))
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}
