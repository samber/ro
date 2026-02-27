//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"context"
	"math"
	"simd/archsimd"

	"github.com/samber/ro"
)

// AVX-512 Int8x64 variants (512-bit vectors)

// AddInt8x64 adds a scalar value to each element of int8x64 vectors.
func AddInt8x64[T ~int8](number T) func(ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
	vector := archsimd.BroadcastInt8x64(int8(number))
	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// SubInt8x64 subtracts a scalar value from each element of int8x64 vectors.
func SubInt8x64[T ~int8](number T) func(ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
	vector := archsimd.BroadcastInt8x64(int8(number))
	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// ReduceSumInt8x64 reduces int8x64 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int8 arithmetic,
// which can only hold values from -128 to 127. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumInt8x64[T ~int8]() func(ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// ClampInt8x64 clamps each element of int8x64 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt8x64[T ~int8](minValue, maxValue T) func(ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt8x64(int8(minValue))
	maxVec := archsimd.BroadcastInt8x64(int8(maxValue))

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// MinInt8x64 applies a minimum bound to each element of int8x64 vectors.
func MinInt8x64[T ~int8](minValue T) func(ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
	minVec := archsimd.BroadcastInt8x64(int8(minValue))

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// MaxInt8x64 applies a maximum bound to each element of int8x64 vectors.
func MaxInt8x64[T ~int8](maxValue T) func(ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
	maxVec := archsimd.BroadcastInt8x64(int8(maxValue))

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// ReduceMinInt8x64 reduces int8x64 vectors to their minimum value.
func ReduceMinInt8x64[T ~int8]() func(ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// ReduceMaxInt8x64 reduces int8x64 vectors to their maximum value.
func ReduceMaxInt8x64[T ~int8]() func(ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
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

// Float32x16 AVX-512 operators (most commonly used AVX-512 type)

// AddFloat32x16 adds a scalar value to each element of float32x16 vectors.
func AddFloat32x16[T ~float32](number T) func(ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
	vector := archsimd.BroadcastFloat32x16(float32(number))
	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// SubFloat32x16 subtracts a scalar value from each element of float32x16 vectors.
func SubFloat32x16[T ~float32](number T) func(ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
	vector := archsimd.BroadcastFloat32x16(float32(number))
	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// ReduceSumFloat32x16 reduces float32x16 vectors to their sum.
func ReduceSumFloat32x16[T ~float32]() func(ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// ClampFloat32x16 clamps each element of float32x16 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampFloat32x16[T ~float32](minValue, maxValue T) func(ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastFloat32x16(float32(minValue))
	maxVec := archsimd.BroadcastFloat32x16(float32(maxValue))

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// MinFloat32x16 applies a minimum bound to each element of float32x16 vectors.
func MinFloat32x16[T ~float32](minValue T) func(ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
	minVec := archsimd.BroadcastFloat32x16(float32(minValue))

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// MaxFloat32x16 applies a maximum bound to each element of float32x16 vectors.
func MaxFloat32x16[T ~float32](maxValue T) func(ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
	maxVec := archsimd.BroadcastFloat32x16(float32(maxValue))

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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

// ReduceMinFloat32x16 reduces float32x16 vectors to their minimum value.
func ReduceMinFloat32x16[T ~float32]() func(ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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
							if math.IsNaN(float64(buf[i])) {
								result = buf[i]
								break
							}
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

// ReduceMaxFloat32x16 reduces float32x16 vectors to their maximum value.
func ReduceMaxFloat32x16[T ~float32]() func(ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
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
							if math.IsNaN(float64(buf[i])) {
								result = buf[i]
								break
							}
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

// Int16x32 AVX-512 operators

// AddInt16x32 adds a scalar value to each element of int16x32 vectors.
func AddInt16x32[T ~int16](number T) func(ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
	vector := archsimd.BroadcastInt16x32(int16(number))
	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// SubInt16x32 subtracts a scalar value from each element of int16x32 vectors.
func SubInt16x32[T ~int16](number T) func(ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
	vector := archsimd.BroadcastInt16x32(int16(number))
	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// ClampInt16x32 clamps each element of int16x32 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt16x32[T ~int16](minValue, maxValue T) func(ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt16x32(int16(minValue))
	maxVec := archsimd.BroadcastInt16x32(int16(maxValue))

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// MinInt16x32 applies a minimum bound to each element of int16x32 vectors.
func MinInt16x32[T ~int16](minValue T) func(ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
	minVec := archsimd.BroadcastInt16x32(int16(minValue))

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// MaxInt16x32 applies a maximum bound to each element of int16x32 vectors.
func MaxInt16x32[T ~int16](maxValue T) func(ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
	maxVec := archsimd.BroadcastInt16x32(int16(maxValue))

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// ReduceSumInt16x32 reduces int16x32 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int16 arithmetic,
// which can only hold values from -32768 to 32767. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumInt16x32[T ~int16]() func(ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// ReduceMinInt16x32 reduces int16x32 vectors to their minimum value.
func ReduceMinInt16x32[T ~int16]() func(ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// ReduceMaxInt16x32 reduces int16x32 vectors to their maximum value.
func ReduceMaxInt16x32[T ~int16]() func(ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
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

// Int32x16 AVX-512 operators

// AddInt32x16 adds a scalar value to each element of int32x16 vectors.
func AddInt32x16[T ~int32](number T) func(ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
	vector := archsimd.BroadcastInt32x16(int32(number))
	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// SubInt32x16 subtracts a scalar value from each element of int32x16 vectors.
func SubInt32x16[T ~int32](number T) func(ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
	vector := archsimd.BroadcastInt32x16(int32(number))
	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// ClampInt32x16 clamps each element of int32x16 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt32x16[T ~int32](minValue, maxValue T) func(ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt32x16(int32(minValue))
	maxVec := archsimd.BroadcastInt32x16(int32(maxValue))

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// MinInt32x16 applies a minimum bound to each element of int32x16 vectors.
func MinInt32x16[T ~int32](minValue T) func(ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
	minVec := archsimd.BroadcastInt32x16(int32(minValue))

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// MaxInt32x16 applies a maximum bound to each element of int32x16 vectors.
func MaxInt32x16[T ~int32](maxValue T) func(ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
	maxVec := archsimd.BroadcastInt32x16(int32(maxValue))

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// ReduceSumInt32x16 reduces int32x16 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int32 arithmetic.
// For extremely large sums that may exceed int32 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumInt32x16[T ~int32]() func(ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// ReduceMinInt32x16 reduces int32x16 vectors to their minimum value.
func ReduceMinInt32x16[T ~int32]() func(ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// ReduceMaxInt32x16 reduces int32x16 vectors to their maximum value.
func ReduceMaxInt32x16[T ~int32]() func(ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
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

// Int64x8 AVX-512 operators

// AddInt64x8 adds a scalar value to each element of int64x8 vectors.
func AddInt64x8[T ~int64](number T) func(ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
	vector := archsimd.BroadcastInt64x8(int64(number))
	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// SubInt64x8 subtracts a scalar value from each element of int64x8 vectors.
func SubInt64x8[T ~int64](number T) func(ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
	vector := archsimd.BroadcastInt64x8(int64(number))
	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// ClampInt64x8 clamps each element of int64x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt64x8[T ~int64](minValue, maxValue T) func(ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt64x8(int64(minValue))
	maxVec := archsimd.BroadcastInt64x8(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// MinInt64x8 applies a minimum bound to each element of int64x8 vectors.
func MinInt64x8[T ~int64](minValue T) func(ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
	minVec := archsimd.BroadcastInt64x8(int64(minValue))

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// MaxInt64x8 applies a maximum bound to each element of int64x8 vectors.
func MaxInt64x8[T ~int64](maxValue T) func(ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
	maxVec := archsimd.BroadcastInt64x8(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// ReduceSumInt64x8 reduces int64x8 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using int64 arithmetic.
// For extremely large sums that may exceed int64 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumInt64x8[T ~int64]() func(ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

// ReduceMinInt64x8 reduces int64x8 vectors to their minimum value.
func ReduceMinInt64x8[T ~int64]() func(ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

						var buf [lanes]int64
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

// ReduceMaxInt64x8 reduces int64x8 vectors to their maximum value.
func ReduceMaxInt64x8[T ~int64]() func(ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
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

						var buf [lanes]int64
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

// ClampInt64x4 clamps each element of int64x4 vectors between min and max values.
// Placed in math_avx512.go because archsimd.Int64x4.Min/Max require AVX-512 (no int64 min/max in AVX2).
func ClampInt64x4[T ~int64](minValue, maxValue T) func(ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt64x4(int64(minValue))
	maxVec := archsimd.BroadcastInt64x4(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
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

// MinInt64x4 applies a minimum bound to each element of int64x4 vectors.
func MinInt64x4[T ~int64](minValue T) func(ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
	minVec := archsimd.BroadcastInt64x4(int64(minValue))

	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
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

// MaxInt64x4 applies a maximum bound to each element of int64x4 vectors.
func MaxInt64x4[T ~int64](maxValue T) func(ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
	maxVec := archsimd.BroadcastInt64x4(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x4]) ro.Observable[*archsimd.Int64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x4) {
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

// ReduceMinInt64x4 reduces int64x4 vectors to their minimum value.
// Placed in math_avx512.go because archsimd.Int64x4.Min requires AVX-512 (no int64 min in AVX2).
func ReduceMinInt64x4[T ~int64]() func(ro.Observable[*archsimd.Int64x4]) ro.Observable[T] {
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

						var buf [lanes]int64
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

// ReduceMaxInt64x4 reduces int64x4 vectors to their maximum value.
// Placed in math_avx512.go because archsimd.Int64x4.Max requires AVX-512 (no int64 max in AVX2).
func ReduceMaxInt64x4[T ~int64]() func(ro.Observable[*archsimd.Int64x4]) ro.Observable[T] {
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

						var buf [lanes]int64
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

// Uint8x64 AVX-512 operators

// AddUint8x64 adds a scalar value to each element of uint8x64 vectors.
func AddUint8x64[T ~uint8](number T) func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
	vector := archsimd.BroadcastUint8x64(uint8(number))
	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// SubUint8x64 subtracts a scalar value from each element of uint8x64 vectors.
func SubUint8x64[T ~uint8](number T) func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
	vector := archsimd.BroadcastUint8x64(uint8(number))
	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// ClampUint8x64 clamps each element of uint8x64 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint8x64[T ~uint8](minValue, maxValue T) func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint8x64(uint8(minValue))
	maxVec := archsimd.BroadcastUint8x64(uint8(maxValue))

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// MinUint8x64 applies a minimum bound to each element of uint8x64 vectors.
func MinUint8x64[T ~uint8](minValue T) func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
	minVec := archsimd.BroadcastUint8x64(uint8(minValue))

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// MaxUint8x64 applies a maximum bound to each element of uint8x64 vectors.
func MaxUint8x64[T ~uint8](maxValue T) func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
	maxVec := archsimd.BroadcastUint8x64(uint8(maxValue))

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// ReduceSumUint8x64 reduces uint8x64 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint8 arithmetic,
// which can only hold values from 0 to 255. For large sums, consider using a wider
// type or a different reduction strategy.
func ReduceSumUint8x64[T ~uint8]() func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// ReduceMinUint8x64 reduces uint8x64 vectors to their minimum value.
func ReduceMinUint8x64[T ~uint8]() func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// ReduceMaxUint8x64 reduces uint8x64 vectors to their maximum value.
func ReduceMaxUint8x64[T ~uint8]() func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint8x64
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
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

// Uint16x32 AVX-512 operators

// AddUint16x32 adds a scalar value to each element of uint16x32 vectors.
func AddUint16x32[T ~uint16](number T) func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
	vector := archsimd.BroadcastUint16x32(uint16(number))
	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// SubUint16x32 subtracts a scalar value from each element of uint16x32 vectors.
func SubUint16x32[T ~uint16](number T) func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
	vector := archsimd.BroadcastUint16x32(uint16(number))
	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// ClampUint16x32 clamps each element of uint16x32 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint16x32[T ~uint16](minValue, maxValue T) func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint16x32(uint16(minValue))
	maxVec := archsimd.BroadcastUint16x32(uint16(maxValue))

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// MinUint16x32 applies a minimum bound to each element of uint16x32 vectors.
func MinUint16x32[T ~uint16](minValue T) func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
	minVec := archsimd.BroadcastUint16x32(uint16(minValue))

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// MaxUint16x32 applies a maximum bound to each element of uint16x32 vectors.
func MaxUint16x32[T ~uint16](maxValue T) func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
	maxVec := archsimd.BroadcastUint16x32(uint16(maxValue))

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// ReduceSumUint16x32 reduces uint16x32 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint16 arithmetic.
// For extremely large sums that may exceed uint16 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint16x32[T ~uint16]() func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// ReduceMinUint16x32 reduces uint16x32 vectors to their minimum value.
func ReduceMinUint16x32[T ~uint16]() func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// ReduceMaxUint16x32 reduces uint16x32 vectors to their maximum value.
func ReduceMaxUint16x32[T ~uint16]() func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint16x32
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
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

// Uint32x16 AVX-512 operators

// AddUint32x16 adds a scalar value to each element of uint32x16 vectors.
func AddUint32x16[T ~uint32](number T) func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
	vector := archsimd.BroadcastUint32x16(uint32(number))
	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// SubUint32x16 subtracts a scalar value from each element of uint32x16 vectors.
func SubUint32x16[T ~uint32](number T) func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
	vector := archsimd.BroadcastUint32x16(uint32(number))
	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// ClampUint32x16 clamps each element of uint32x16 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint32x16[T ~uint32](minValue, maxValue T) func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint32x16(uint32(minValue))
	maxVec := archsimd.BroadcastUint32x16(uint32(maxValue))

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// MinUint32x16 applies a minimum bound to each element of uint32x16 vectors.
func MinUint32x16[T ~uint32](minValue T) func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
	minVec := archsimd.BroadcastUint32x16(uint32(minValue))

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// MaxUint32x16 applies a maximum bound to each element of uint32x16 vectors.
func MaxUint32x16[T ~uint32](maxValue T) func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
	maxVec := archsimd.BroadcastUint32x16(uint32(maxValue))

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// ReduceSumUint32x16 reduces uint32x16 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint32 arithmetic.
// For extremely large sums that may exceed uint32 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint32x16[T ~uint32]() func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// ReduceMinUint32x16 reduces uint32x16 vectors to their minimum value.
func ReduceMinUint32x16[T ~uint32]() func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// ReduceMaxUint32x16 reduces uint32x16 vectors to their maximum value.
func ReduceMaxUint32x16[T ~uint32]() func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint32x16
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
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

// Uint64x8 AVX-512 operators

// AddUint64x8 adds a scalar value to each element of uint64x8 vectors.
func AddUint64x8[T ~uint64](number T) func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
	vector := archsimd.BroadcastUint64x8(uint64(number))
	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// SubUint64x8 subtracts a scalar value from each element of uint64x8 vectors.
func SubUint64x8[T ~uint64](number T) func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
	vector := archsimd.BroadcastUint64x8(uint64(number))
	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// ClampUint64x8 clamps each element of uint64x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint64x8[T ~uint64](minValue, maxValue T) func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint64x8(uint64(minValue))
	maxVec := archsimd.BroadcastUint64x8(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// MinUint64x8 applies a minimum bound to each element of uint64x8 vectors.
func MinUint64x8[T ~uint64](minValue T) func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
	minVec := archsimd.BroadcastUint64x8(uint64(minValue))

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// MaxUint64x8 applies a maximum bound to each element of uint64x8 vectors.
func MaxUint64x8[T ~uint64](maxValue T) func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
	maxVec := archsimd.BroadcastUint64x8(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// ReduceSumUint64x8 reduces uint64x8 vectors to their sum.
//
// WARNING: This function may overflow. The result is accumulated using uint64 arithmetic.
// For extremely large sums that may exceed uint64 range, consider using a wider type
// or a different reduction strategy.
func ReduceSumUint64x8[T ~uint64]() func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

// ReduceMinUint64x8 reduces uint64x8 vectors to their minimum value.
func ReduceMinUint64x8[T ~uint64]() func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

						var buf [lanes]uint64
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

// ReduceMaxUint64x8 reduces uint64x8 vectors to their maximum value.
func ReduceMaxUint64x8[T ~uint64]() func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
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

						var buf [lanes]uint64
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

// ClampUint64x4 clamps each element of uint64x4 vectors between min and max values.
// Placed in math_avx512.go because archsimd.Uint64x4.Min/Max require AVX-512 (no uint64 min/max in AVX2).
func ClampUint64x4[T ~uint64](minValue, maxValue T) func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint64x4(uint64(minValue))
	maxVec := archsimd.BroadcastUint64x4(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
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

// MinUint64x4 applies a minimum bound to each element of uint64x4 vectors.
func MinUint64x4[T ~uint64](minValue T) func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
	minVec := archsimd.BroadcastUint64x4(uint64(minValue))

	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
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

// MaxUint64x4 applies a maximum bound to each element of uint64x4 vectors.
func MaxUint64x4[T ~uint64](maxValue T) func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
	maxVec := archsimd.BroadcastUint64x4(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x4]) ro.Observable[*archsimd.Uint64x4] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x4]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x4) {
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

// ReduceMinUint64x4 reduces uint64x4 vectors to their minimum value.
// Placed in math_avx512.go because archsimd.Uint64x4.Min requires AVX-512 (no uint64 min in AVX2).
func ReduceMinUint64x4[T ~uint64]() func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[T] {
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

						var buf [lanes]uint64
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

// ReduceMaxUint64x4 reduces uint64x4 vectors to their maximum value.
// Placed in math_avx512.go because archsimd.Uint64x4.Max requires AVX-512 (no uint64 max in AVX2).
func ReduceMaxUint64x4[T ~uint64]() func(ro.Observable[*archsimd.Uint64x4]) ro.Observable[T] {
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

						var buf [lanes]uint64
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

// Float64x8 AVX-512 operators

// AddFloat64x8 adds a scalar value to each element of float64x8 vectors.
func AddFloat64x8[T ~float64](number T) func(ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
	vector := archsimd.BroadcastFloat64x8(float64(number))
	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// SubFloat64x8 subtracts a scalar value from each element of float64x8 vectors.
func SubFloat64x8[T ~float64](number T) func(ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
	vector := archsimd.BroadcastFloat64x8(float64(number))
	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// ClampFloat64x8 clamps each element of float64x8 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampFloat64x8[T ~float64](minValue, maxValue T) func(ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastFloat64x8(float64(minValue))
	maxVec := archsimd.BroadcastFloat64x8(float64(maxValue))

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// MinFloat64x8 applies a minimum bound to each element of float64x8 vectors.
func MinFloat64x8[T ~float64](minValue T) func(ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
	minVec := archsimd.BroadcastFloat64x8(float64(minValue))

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// MaxFloat64x8 applies a maximum bound to each element of float64x8 vectors.
func MaxFloat64x8[T ~float64](maxValue T) func(ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
	maxVec := archsimd.BroadcastFloat64x8(float64(maxValue))

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// ReduceSumFloat64x8 reduces float64x8 vectors to their sum.
func ReduceSumFloat64x8[T ~float64]() func(ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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

// ReduceMinFloat64x8 reduces float64x8 vectors to their minimum value.
func ReduceMinFloat64x8[T ~float64]() func(ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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
							result = math.Min(result, buf[i])
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

// ReduceMaxFloat64x8 reduces float64x8 vectors to their maximum value.
func ReduceMaxFloat64x8[T ~float64]() func(ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Float64x8
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
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
							result = math.Max(result, buf[i])
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

// ==================== Int64x2 and Uint64x2 Min/Max/Clamp functions ====================
// NOTE: These operations require AVX-512 because AVX and AVX2 don't have 64-bit integer comparison instructions

// ClampInt64x2 clamps each element of int64x2 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampInt64x2[T ~int64](minValue, maxValue T) func(ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastInt64x2(int64(minValue))
	maxVec := archsimd.BroadcastInt64x2(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x2) {
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

// MinInt64x2 applies a minimum bound to each element of int64x2 vectors.
func MinInt64x2[T ~int64](minValue T) func(ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
	minVec := archsimd.BroadcastInt64x2(int64(minValue))

	return func(source ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x2) {
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

// MaxInt64x2 applies a maximum bound to each element of int64x2 vectors.
func MaxInt64x2[T ~int64](maxValue T) func(ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
	maxVec := archsimd.BroadcastInt64x2(int64(maxValue))

	return func(source ro.Observable[*archsimd.Int64x2]) ro.Observable[*archsimd.Int64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x2) {
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

// ReduceMinInt64x2 reduces int64x2 vectors to their minimum value.
func ReduceMinInt64x2[T ~int64]() func(ro.Observable[*archsimd.Int64x2]) ro.Observable[T] {
	const lanes = simdLanes2

	return func(source ro.Observable[*archsimd.Int64x2]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x2
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x2) {
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

						var buf [lanes]int64
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

// ReduceMaxInt64x2 reduces int64x2 vectors to their maximum value.
func ReduceMaxInt64x2[T ~int64]() func(ro.Observable[*archsimd.Int64x2]) ro.Observable[T] {
	const lanes = simdLanes2

	return func(source ro.Observable[*archsimd.Int64x2]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Int64x2
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x2) {
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

						var buf [lanes]int64
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

// ClampUint64x2 clamps each element of uint64x2 vectors between min and max values.
// Values outside the range are clamped to the nearest valid value.
func ClampUint64x2[T ~uint64](minValue, maxValue T) func(ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
	if minValue > maxValue {
		panic(ErrClampLowerLessThanUpper)
	}

	minVec := archsimd.BroadcastUint64x2(uint64(minValue))
	maxVec := archsimd.BroadcastUint64x2(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x2) {
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

// MinUint64x2 applies a minimum bound to each element of uint64x2 vectors.
func MinUint64x2[T ~uint64](minValue T) func(ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
	minVec := archsimd.BroadcastUint64x2(uint64(minValue))

	return func(source ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x2) {
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

// MaxUint64x2 applies a maximum bound to each element of uint64x2 vectors.
func MaxUint64x2[T ~uint64](maxValue T) func(ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
	maxVec := archsimd.BroadcastUint64x2(uint64(maxValue))

	return func(source ro.Observable[*archsimd.Uint64x2]) ro.Observable[*archsimd.Uint64x2] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x2]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x2) {
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

// ReduceMinUint64x2 reduces uint64x2 vectors to their minimum value.
func ReduceMinUint64x2[T ~uint64]() func(ro.Observable[*archsimd.Uint64x2]) ro.Observable[T] {
	const lanes = simdLanes2

	return func(source ro.Observable[*archsimd.Uint64x2]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x2
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x2) {
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

						var buf [lanes]uint64
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

// ReduceMaxUint64x2 reduces uint64x2 vectors to their maximum value.
func ReduceMaxUint64x2[T ~uint64]() func(ro.Observable[*archsimd.Uint64x2]) ro.Observable[T] {
	const lanes = simdLanes2

	return func(source ro.Observable[*archsimd.Uint64x2]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			var accumulation archsimd.Uint64x2
			var initialized bool

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x2) {
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

						var buf [lanes]uint64
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
