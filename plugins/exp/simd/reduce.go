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

package rosimd

import "github.com/samber/ro"

// Reductions that fold every valid lane of a stream into one scalar: ReduceSum,
// ReduceMin and ReduceMax.
//
// simd offers no horizontal reduction, so collapsing lanes always means storing the
// vector and folding in scalar code. Each of these delegates to reduceLanes, which does
// that once per vector.

// ReduceSumInt8 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in int8 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[int8, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.ReduceSumInt8,
//	)
func ReduceSumInt8[V Int8Vector[V]](source ro.Observable[V]) ro.Observable[int8] {
	return reduceLanes(
		source,
		func(acc, lane int8) int8 { return acc + lane },
		func(acc int8, _ bool, emit func(int8)) { emit(acc) },
	)
}

// ReduceSumInt16 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in int16 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[int16, rosimd.PartialInt16s, int16](
//		source,
//		rosimd.VectorizeInt16[rosimd.PartialInt16s](),
//		rosimd.ReduceSumInt16,
//	)
func ReduceSumInt16[V Int16Vector[V]](source ro.Observable[V]) ro.Observable[int16] {
	return reduceLanes(
		source,
		func(acc, lane int16) int16 { return acc + lane },
		func(acc int16, _ bool, emit func(int16)) { emit(acc) },
	)
}

// ReduceSumInt32 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in int32 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[int32, rosimd.PartialInt32s, int32](
//		source,
//		rosimd.VectorizeInt32[rosimd.PartialInt32s](),
//		rosimd.ReduceSumInt32,
//	)
func ReduceSumInt32[V Int32Vector[V]](source ro.Observable[V]) ro.Observable[int32] {
	return reduceLanes(
		source,
		func(acc, lane int32) int32 { return acc + lane },
		func(acc int32, _ bool, emit func(int32)) { emit(acc) },
	)
}

// ReduceSumInt64 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in int64 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[int64, rosimd.PartialInt64s, int64](
//		source,
//		rosimd.VectorizeInt64[rosimd.PartialInt64s](),
//		rosimd.ReduceSumInt64,
//	)
func ReduceSumInt64[V Int64Vector[V]](source ro.Observable[V]) ro.Observable[int64] {
	return reduceLanes(
		source,
		func(acc, lane int64) int64 { return acc + lane },
		func(acc int64, _ bool, emit func(int64)) { emit(acc) },
	)
}

// ReduceSumUint8 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in uint8 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[uint8, rosimd.PartialUint8s, uint8](
//		source,
//		rosimd.VectorizeUint8[rosimd.PartialUint8s](),
//		rosimd.ReduceSumUint8,
//	)
func ReduceSumUint8[V Uint8Vector[V]](source ro.Observable[V]) ro.Observable[uint8] {
	return reduceLanes(
		source,
		func(acc, lane uint8) uint8 { return acc + lane },
		func(acc uint8, _ bool, emit func(uint8)) { emit(acc) },
	)
}

// ReduceSumUint16 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in uint16 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[uint16, rosimd.PartialUint16s, uint16](
//		source,
//		rosimd.VectorizeUint16[rosimd.PartialUint16s](),
//		rosimd.ReduceSumUint16,
//	)
func ReduceSumUint16[V Uint16Vector[V]](source ro.Observable[V]) ro.Observable[uint16] {
	return reduceLanes(
		source,
		func(acc, lane uint16) uint16 { return acc + lane },
		func(acc uint16, _ bool, emit func(uint16)) { emit(acc) },
	)
}

// ReduceSumUint32 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in uint32 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[uint32, rosimd.PartialUint32s, uint32](
//		source,
//		rosimd.VectorizeUint32[rosimd.PartialUint32s](),
//		rosimd.ReduceSumUint32,
//	)
func ReduceSumUint32[V Uint32Vector[V]](source ro.Observable[V]) ro.Observable[uint32] {
	return reduceLanes(
		source,
		func(acc, lane uint32) uint32 { return acc + lane },
		func(acc uint32, _ bool, emit func(uint32)) { emit(acc) },
	)
}

// ReduceSumUint64 totals every valid lane of the stream and emits the sum on completion.
//
// The sum wraps on overflow, exactly as ro.Sum does — it accumulates in uint64 rather
// than promoting to a wider type. An empty stream emits zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[uint64, rosimd.PartialUint64s, uint64](
//		source,
//		rosimd.VectorizeUint64[rosimd.PartialUint64s](),
//		rosimd.ReduceSumUint64,
//	)
func ReduceSumUint64[V Uint64Vector[V]](source ro.Observable[V]) ro.Observable[uint64] {
	return reduceLanes(
		source,
		func(acc, lane uint64) uint64 { return acc + lane },
		func(acc uint64, _ bool, emit func(uint64)) { emit(acc) },
	)
}

// ReduceSumFloat32 totals every valid lane of the stream and emits the sum on
// completion.
//
// It accumulates in float32 and adds lane by lane in stream order, exactly as ro.Sum
// does, so the rounding is identical rather than merely close. An empty stream emits
// zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[float32, rosimd.PartialFloat32s, float32](
//		source,
//		rosimd.VectorizeFloat32[rosimd.PartialFloat32s](),
//		rosimd.ReduceSumFloat32,
//	)
func ReduceSumFloat32[V Float32Vector[V]](source ro.Observable[V]) ro.Observable[float32] {
	return reduceLanes(
		source,
		func(acc, lane float32) float32 { return acc + lane },
		func(acc float32, _ bool, emit func(float32)) { emit(acc) },
	)
}

// ReduceSumFloat64 totals every valid lane of the stream and emits the sum on
// completion.
//
// It accumulates in float64 and adds lane by lane in stream order, exactly as ro.Sum
// does, so the rounding is identical rather than merely close. An empty stream emits
// zero.
//
// It is not a curried operator, so the type argument is inferred from the Pipe:
//
//	ro.Pipe2[float64, rosimd.PartialFloat64s, float64](
//		source,
//		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
//		rosimd.ReduceSumFloat64,
//	)
func ReduceSumFloat64[V Float64Vector[V]](source ro.Observable[V]) ro.Observable[float64] {
	return reduceLanes(
		source,
		func(acc, lane float64) float64 { return acc + lane },
		func(acc float64, _ bool, emit func(float64)) { emit(acc) },
	)
}

// ReduceMinInt8 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinInt8[V Int8Vector[V]](source ro.Observable[V]) ro.Observable[int8] {
	return reduceLanes(
		source,
		func(acc, lane int8) int8 { return min(acc, lane) },
		emitWhenSeen[int8],
	)
}

// ReduceMinInt16 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinInt16[V Int16Vector[V]](source ro.Observable[V]) ro.Observable[int16] {
	return reduceLanes(
		source,
		func(acc, lane int16) int16 { return min(acc, lane) },
		emitWhenSeen[int16],
	)
}

// ReduceMinInt32 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinInt32[V Int32Vector[V]](source ro.Observable[V]) ro.Observable[int32] {
	return reduceLanes(
		source,
		func(acc, lane int32) int32 { return min(acc, lane) },
		emitWhenSeen[int32],
	)
}

// ReduceMinInt64 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinInt64[V Int64Vector[V]](source ro.Observable[V]) ro.Observable[int64] {
	return reduceLanes(
		source,
		func(acc, lane int64) int64 { return min(acc, lane) },
		emitWhenSeen[int64],
	)
}

// ReduceMinUint8 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinUint8[V Uint8Vector[V]](source ro.Observable[V]) ro.Observable[uint8] {
	return reduceLanes(
		source,
		func(acc, lane uint8) uint8 { return min(acc, lane) },
		emitWhenSeen[uint8],
	)
}

// ReduceMinUint16 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinUint16[V Uint16Vector[V]](source ro.Observable[V]) ro.Observable[uint16] {
	return reduceLanes(
		source,
		func(acc, lane uint16) uint16 { return min(acc, lane) },
		emitWhenSeen[uint16],
	)
}

// ReduceMinUint32 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinUint32[V Uint32Vector[V]](source ro.Observable[V]) ro.Observable[uint32] {
	return reduceLanes(
		source,
		func(acc, lane uint32) uint32 { return min(acc, lane) },
		emitWhenSeen[uint32],
	)
}

// ReduceMinUint64 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
func ReduceMinUint64[V Uint64Vector[V]](source ro.Observable[V]) ro.Observable[uint64] {
	return reduceLanes(
		source,
		func(acc, lane uint64) uint64 { return min(acc, lane) },
		emitWhenSeen[uint64],
	)
}

// ReduceMinFloat32 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
//
// NaN is handled the opposite way round from the element-wise MinFloat32. The
// comparison is a plain <, false for any NaN, so a NaN lane never displaces the
// accumulator — which is what ro.Min does, and the point of this operator is to agree
// with it. A stream whose very first lane is NaN still reduces to NaN, again matching
// ro.Min, because nothing can compare less than it.
func ReduceMinFloat32[V Float32Vector[V]](source ro.Observable[V]) ro.Observable[float32] {
	return reduceLanes(
		source,
		func(acc, lane float32) float32 {
			if lane < acc {
				return lane
			}

			return acc
		},
		emitWhenSeen[float32],
	)
}

// ReduceMinFloat64 emits the smallest valid lane of the stream on completion.
//
// An empty stream emits nothing, matching ro.Min.
//
// NaN is handled the opposite way round from the element-wise MinFloat64. The
// comparison is a plain <, false for any NaN, so a NaN lane never displaces the
// accumulator — which is what ro.Min does, and the point of this operator is to agree
// with it. A stream whose very first lane is NaN still reduces to NaN, again matching
// ro.Min, because nothing can compare less than it.
func ReduceMinFloat64[V Float64Vector[V]](source ro.Observable[V]) ro.Observable[float64] {
	return reduceLanes(
		source,
		func(acc, lane float64) float64 {
			if lane < acc {
				return lane
			}

			return acc
		},
		emitWhenSeen[float64],
	)
}

// ReduceMaxInt8 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxInt8[V Int8Vector[V]](source ro.Observable[V]) ro.Observable[int8] {
	return reduceLanes(
		source,
		func(acc, lane int8) int8 { return max(acc, lane) },
		emitWhenSeen[int8],
	)
}

// ReduceMaxInt16 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxInt16[V Int16Vector[V]](source ro.Observable[V]) ro.Observable[int16] {
	return reduceLanes(
		source,
		func(acc, lane int16) int16 { return max(acc, lane) },
		emitWhenSeen[int16],
	)
}

// ReduceMaxInt32 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxInt32[V Int32Vector[V]](source ro.Observable[V]) ro.Observable[int32] {
	return reduceLanes(
		source,
		func(acc, lane int32) int32 { return max(acc, lane) },
		emitWhenSeen[int32],
	)
}

// ReduceMaxInt64 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxInt64[V Int64Vector[V]](source ro.Observable[V]) ro.Observable[int64] {
	return reduceLanes(
		source,
		func(acc, lane int64) int64 { return max(acc, lane) },
		emitWhenSeen[int64],
	)
}

// ReduceMaxUint8 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxUint8[V Uint8Vector[V]](source ro.Observable[V]) ro.Observable[uint8] {
	return reduceLanes(
		source,
		func(acc, lane uint8) uint8 { return max(acc, lane) },
		emitWhenSeen[uint8],
	)
}

// ReduceMaxUint16 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxUint16[V Uint16Vector[V]](source ro.Observable[V]) ro.Observable[uint16] {
	return reduceLanes(
		source,
		func(acc, lane uint16) uint16 { return max(acc, lane) },
		emitWhenSeen[uint16],
	)
}

// ReduceMaxUint32 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxUint32[V Uint32Vector[V]](source ro.Observable[V]) ro.Observable[uint32] {
	return reduceLanes(
		source,
		func(acc, lane uint32) uint32 { return max(acc, lane) },
		emitWhenSeen[uint32],
	)
}

// ReduceMaxUint64 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing.
func ReduceMaxUint64[V Uint64Vector[V]](source ro.Observable[V]) ro.Observable[uint64] {
	return reduceLanes(
		source,
		func(acc, lane uint64) uint64 { return max(acc, lane) },
		emitWhenSeen[uint64],
	)
}

// ReduceMaxFloat32 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing. NaN never displaces the accumulator, matching ro.Max
// — see ReduceMinFloat32 for why this differs from the element-wise MaxFloat32.
func ReduceMaxFloat32[V Float32Vector[V]](source ro.Observable[V]) ro.Observable[float32] {
	return reduceLanes(
		source,
		func(acc, lane float32) float32 {
			if lane > acc {
				return lane
			}

			return acc
		},
		emitWhenSeen[float32],
	)
}

// ReduceMaxFloat64 emits the largest valid lane of the stream on completion.
//
// An empty stream emits nothing. NaN never displaces the accumulator, matching ro.Max
// — see ReduceMinFloat64 for why this differs from the element-wise MaxFloat64.
func ReduceMaxFloat64[V Float64Vector[V]](source ro.Observable[V]) ro.Observable[float64] {
	return reduceLanes(
		source,
		func(acc, lane float64) float64 {
			if lane > acc {
				return lane
			}

			return acc
		},
		emitWhenSeen[float64],
	)
}
