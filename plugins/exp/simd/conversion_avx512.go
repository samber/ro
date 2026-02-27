//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"context"

	"simd/archsimd"

	"github.com/samber/ro"
)

// AVX-512 To/From conversion helpers (512-bit vectors)

// Int8x64 AVX-512 variants

// ScalarToInt8x64 converts a stream of scalar int8 values into int8x64 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 64 elements. When the buffer
// is full (exactly 64 values), a complete Int8x64 vector is emitted. On source completion,
// any remaining values in the buffer (less than 64) are emitted as a partial vector.
func ScalarToInt8x64[T ~int8]() func(ro.Observable[T]) ro.Observable[*archsimd.Int8x64] {
	const lanes = simdLanes64

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Int8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int8x64]) ro.Teardown {
			var buf [lanes]int8
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = int8(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadInt8x64(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]int8{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadInt8x64(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int8x64ToScalar converts a stream of int8x64 vectors into scalar int8 values.
//
// Buffer semantics: Each Int8x64 vector is fully unpacked into 64 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 64 elements set) will emit zero values for unused lanes.
func Int8x64ToScalar[T ~int8]() func(ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Int8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int8x64) {
						var buf [lanes]int8
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Float32x16 AVX-512 variants

// ScalarToFloat32x16 converts a stream of scalar float32 values into float32x16 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 16 elements. When the buffer
// is full (exactly 16 values), a complete Float32x16 vector is emitted. On source completion,
// any remaining values in the buffer (less than 16) are emitted as a partial vector.
func ScalarToFloat32x16[T ~float32]() func(ro.Observable[T]) ro.Observable[*archsimd.Float32x16] {
	const lanes = simdLanes16

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Float32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float32x16]) ro.Teardown {
			var buf [lanes]float32
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = float32(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadFloat32x16(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]float32{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadFloat32x16(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Float32x16ToScalar converts a stream of float32x16 vectors into scalar float32 values.
//
// Buffer semantics: Each Float32x16 vector is fully unpacked into 16 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 16 elements set) will emit zero values for unused lanes.
func Float32x16ToScalar[T ~float32]() func(ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Float32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float32x16) {
						var buf [lanes]float32
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Int16x32 AVX-512 variants

// ScalarToInt16x32 converts a stream of scalar int16 values into int16x32 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 32 elements. When the buffer
// is full (exactly 32 values), a complete Int16x32 vector is emitted. On source completion,
// any remaining values in the buffer (less than 32) are emitted as a partial vector.
func ScalarToInt16x32[T ~int16]() func(ro.Observable[T]) ro.Observable[*archsimd.Int16x32] {
	const lanes = simdLanes32

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Int16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int16x32]) ro.Teardown {
			var buf [lanes]int16
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = int16(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadInt16x32(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]int16{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadInt16x32(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int16x32ToScalar converts a stream of int16x32 vectors into scalar int16 values.
//
// Buffer semantics: Each Int16x32 vector is fully unpacked into 32 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 32 elements set) will emit zero values for unused lanes.
func Int16x32ToScalar[T ~int16]() func(ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Int16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int16x32) {
						var buf [lanes]int16
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Int32x16 AVX-512 variants

// ScalarToInt32x16 converts a stream of scalar int32 values into int32x16 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 16 elements. When the buffer
// is full (exactly 16 values), a complete Int32x16 vector is emitted. On source completion,
// any remaining values in the buffer (less than 16) are emitted as a partial vector.
func ScalarToInt32x16[T ~int32]() func(ro.Observable[T]) ro.Observable[*archsimd.Int32x16] {
	const lanes = simdLanes16

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Int32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int32x16]) ro.Teardown {
			var buf [lanes]int32
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = int32(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadInt32x16(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]int32{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadInt32x16(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int32x16ToScalar converts a stream of int32x16 vectors into scalar int32 values.
//
// Buffer semantics: Each Int32x16 vector is fully unpacked into 16 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 16 elements set) will emit zero values for unused lanes.
func Int32x16ToScalar[T ~int32]() func(ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Int32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int32x16) {
						var buf [lanes]int32
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Int64x8 AVX-512 variants

// ScalarToInt64x8 converts a stream of scalar int64 values into int64x8 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 8 elements. When the buffer
// is full (exactly 8 values), a complete Int64x8 vector is emitted. On source completion,
// any remaining values in the buffer (less than 8) are emitted as a partial vector.
func ScalarToInt64x8[T ~int64]() func(ro.Observable[T]) ro.Observable[*archsimd.Int64x8] {
	const lanes = simdLanes8

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Int64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Int64x8]) ro.Teardown {
			var buf [lanes]int64
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = int64(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadInt64x8(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]int64{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadInt64x8(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Int64x8ToScalar converts a stream of int64x8 vectors into scalar int64 values.
//
// Buffer semantics: Each Int64x8 vector is fully unpacked into 8 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 8 elements set) will emit zero values for unused lanes.
func Int64x8ToScalar[T ~int64]() func(ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Int64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Int64x8) {
						var buf [lanes]int64
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Uint8x64 AVX-512 variants

// ScalarToUint8x64 converts a stream of scalar uint8 values into uint8x64 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 64 elements. When the buffer
// is full (exactly 64 values), a complete Uint8x64 vector is emitted. On source completion,
// any remaining values in the buffer (less than 64) are emitted as a partial vector.
func ScalarToUint8x64[T ~uint8]() func(ro.Observable[T]) ro.Observable[*archsimd.Uint8x64] {
	const lanes = simdLanes64

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Uint8x64] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint8x64]) ro.Teardown {
			var buf [lanes]uint8
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = uint8(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadUint8x64(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]uint8{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadUint8x64(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint8x64ToScalar converts a stream of uint8x64 vectors into scalar uint8 values.
//
// Buffer semantics: Each Uint8x64 vector is fully unpacked into 64 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 64 elements set) will emit zero values for unused lanes.
func Uint8x64ToScalar[T ~uint8]() func(ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
	const lanes = simdLanes64

	return func(source ro.Observable[*archsimd.Uint8x64]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint8x64) {
						var buf [lanes]uint8
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Uint16x32 AVX-512 variants

// ScalarToUint16x32 converts a stream of scalar uint16 values into uint16x32 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 32 elements. When the buffer
// is full (exactly 32 values), a complete Uint16x32 vector is emitted. On source completion,
// any remaining values in the buffer (less than 32) are emitted as a partial vector.
func ScalarToUint16x32[T ~uint16]() func(ro.Observable[T]) ro.Observable[*archsimd.Uint16x32] {
	const lanes = simdLanes32

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Uint16x32] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint16x32]) ro.Teardown {
			var buf [lanes]uint16
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = uint16(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadUint16x32(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]uint16{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadUint16x32(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint16x32ToScalar converts a stream of uint16x32 vectors into scalar uint16 values.
//
// Buffer semantics: Each Uint16x32 vector is fully unpacked into 32 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 32 elements set) will emit zero values for unused lanes.
func Uint16x32ToScalar[T ~uint16]() func(ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
	const lanes = simdLanes32

	return func(source ro.Observable[*archsimd.Uint16x32]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint16x32) {
						var buf [lanes]uint16
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Uint32x16 AVX-512 variants

// ScalarToUint32x16 converts a stream of scalar uint32 values into uint32x16 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 16 elements. When the buffer
// is full (exactly 16 values), a complete Uint32x16 vector is emitted. On source completion,
// any remaining values in the buffer (less than 16) are emitted as a partial vector.
func ScalarToUint32x16[T ~uint32]() func(ro.Observable[T]) ro.Observable[*archsimd.Uint32x16] {
	const lanes = simdLanes16

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Uint32x16] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint32x16]) ro.Teardown {
			var buf [lanes]uint32
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = uint32(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadUint32x16(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]uint32{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadUint32x16(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint32x16ToScalar converts a stream of uint32x16 vectors into scalar uint32 values.
//
// Buffer semantics: Each Uint32x16 vector is fully unpacked into 16 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 16 elements set) will emit zero values for unused lanes.
func Uint32x16ToScalar[T ~uint32]() func(ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
	const lanes = simdLanes16

	return func(source ro.Observable[*archsimd.Uint32x16]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint32x16) {
						var buf [lanes]uint32
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Uint64x8 AVX-512 variants

// ScalarToUint64x8 converts a stream of scalar uint64 values into uint64x8 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 8 elements. When the buffer
// is full (exactly 8 values), a complete Uint64x8 vector is emitted. On source completion,
// any remaining values in the buffer (less than 8) are emitted as a partial vector.
func ScalarToUint64x8[T ~uint64]() func(ro.Observable[T]) ro.Observable[*archsimd.Uint64x8] {
	const lanes = simdLanes8

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Uint64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Uint64x8]) ro.Teardown {
			var buf [lanes]uint64
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = uint64(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadUint64x8(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]uint64{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadUint64x8(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Uint64x8ToScalar converts a stream of uint64x8 vectors into scalar uint64 values.
//
// Buffer semantics: Each Uint64x8 vector is fully unpacked into 8 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 8 elements set) will emit zero values for unused lanes.
func Uint64x8ToScalar[T ~uint64]() func(ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Uint64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Uint64x8) {
						var buf [lanes]uint64
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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

// Float64x8 AVX-512 variants

// ScalarToFloat64x8 converts a stream of scalar float64 values into float64x8 vectors.
//
// Buffer semantics: Values are accumulated into a buffer of 8 elements. When the buffer
// is full (exactly 8 values), a complete Float64x8 vector is emitted. On source completion,
// any remaining values in the buffer (less than 8) are emitted as a partial vector.
func ScalarToFloat64x8[T ~float64]() func(ro.Observable[T]) ro.Observable[*archsimd.Float64x8] {
	const lanes = simdLanes8

	return func(source ro.Observable[T]) ro.Observable[*archsimd.Float64x8] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[*archsimd.Float64x8]) ro.Teardown {
			var buf [lanes]float64
			var index uint

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value T) {
						buf[index] = float64(value)
						index++

						if uint(index) == lanes {
							vec := archsimd.LoadFloat64x8(&buf)
							destination.NextWithContext(ctx, &vec)
							buf = [lanes]float64{}
							index = 0
						}
					},
					destination.ErrorWithContext,
					func(ctx context.Context) {
						if index > 0 {
							vec := archsimd.LoadFloat64x8(&buf)
							destination.NextWithContext(ctx, &vec)
						}
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// Float64x8ToScalar converts a stream of float64x8 vectors into scalar float64 values.
//
// Buffer semantics: Each Float64x8 vector is fully unpacked into 8 scalar values, emitting
// each element individually. All elements from each input vector are emitted in order.
// Partial vectors (fewer than 8 elements set) will emit zero values for unused lanes.
func Float64x8ToScalar[T ~float64]() func(ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
	const lanes = simdLanes8

	return func(source ro.Observable[*archsimd.Float64x8]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value *archsimd.Float64x8) {
						var buf [lanes]float64
						value.Store(&buf)
						for i := uint(0); i < lanes; i++ {
							destination.NextWithContext(ctx, T(buf[i]))
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
