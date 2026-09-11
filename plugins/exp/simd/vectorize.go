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

// Crossing the boundary of vector space, in both directions.
//
// Vectorize batches a scalar stream into vectors. ToScalar and Flatten bring it back out
// — the first as one slice per vector, the second as one value per lane. The Reduce
// operators in reduce.go are the other way out, collapsing a whole stream to one value.
//
// Vectorize produces the Partial types alone, since only they carry the validity mask a
// short final batch needs. The two exits are less demanding: they ask only that a vector
// can report its lanes, so the standard library's vector types work too.

// VectorizeInt8 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialInt8s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[int8, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8,
//		rosimd.ReduceSumInt8,
//	)
func VectorizeInt8[V Int8Buffer[V]](source ro.Observable[int8]) ro.Observable[V] {
	return vectorize[int8, V](source)
}

// VectorizeInt16 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialInt16s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[int16, rosimd.PartialInt16s, int16](
//		source,
//		rosimd.VectorizeInt16,
//		rosimd.ReduceSumInt16,
//	)
func VectorizeInt16[V Int16Buffer[V]](source ro.Observable[int16]) ro.Observable[V] {
	return vectorize[int16, V](source)
}

// VectorizeInt32 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialInt32s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[int32, rosimd.PartialInt32s, int32](
//		source,
//		rosimd.VectorizeInt32,
//		rosimd.ReduceSumInt32,
//	)
func VectorizeInt32[V Int32Buffer[V]](source ro.Observable[int32]) ro.Observable[V] {
	return vectorize[int32, V](source)
}

// VectorizeInt64 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialInt64s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[int64, rosimd.PartialInt64s, int64](
//		source,
//		rosimd.VectorizeInt64,
//		rosimd.ReduceSumInt64,
//	)
func VectorizeInt64[V Int64Buffer[V]](source ro.Observable[int64]) ro.Observable[V] {
	return vectorize[int64, V](source)
}

// VectorizeUint8 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialUint8s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[uint8, rosimd.PartialUint8s, uint8](
//		source,
//		rosimd.VectorizeUint8,
//		rosimd.ReduceSumUint8,
//	)
func VectorizeUint8[V Uint8Buffer[V]](source ro.Observable[uint8]) ro.Observable[V] {
	return vectorize[uint8, V](source)
}

// VectorizeUint16 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialUint16s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[uint16, rosimd.PartialUint16s, uint16](
//		source,
//		rosimd.VectorizeUint16,
//		rosimd.ReduceSumUint16,
//	)
func VectorizeUint16[V Uint16Buffer[V]](source ro.Observable[uint16]) ro.Observable[V] {
	return vectorize[uint16, V](source)
}

// VectorizeUint32 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialUint32s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[uint32, rosimd.PartialUint32s, uint32](
//		source,
//		rosimd.VectorizeUint32,
//		rosimd.ReduceSumUint32,
//	)
func VectorizeUint32[V Uint32Buffer[V]](source ro.Observable[uint32]) ro.Observable[V] {
	return vectorize[uint32, V](source)
}

// VectorizeUint64 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialUint64s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[uint64, rosimd.PartialUint64s, uint64](
//		source,
//		rosimd.VectorizeUint64,
//		rosimd.ReduceSumUint64,
//	)
func VectorizeUint64[V Uint64Buffer[V]](source ro.Observable[uint64]) ro.Observable[V] {
	return vectorize[uint64, V](source)
}

// VectorizeFloat32 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialFloat32s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[float32, rosimd.PartialFloat32s, float32](
//		source,
//		rosimd.VectorizeFloat32,
//		rosimd.ReduceSumFloat32,
//	)
func VectorizeFloat32[V Float32Buffer[V]](source ro.Observable[float32]) ro.Observable[V] {
	return vectorize[float32, V](source)
}

// VectorizeFloat64 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one
// final PartialFloat64s holding whatever is left. Downstream operators see that short
// vector as a first-class value rather than a special case, because its padded lanes
// are masked out of every operation.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe, and avoids the compiler bug that breaks generic
// functions whose only parameter is a type parameter.
//
//	ro.Pipe2[float64, rosimd.PartialFloat64s, float64](
//		source,
//		rosimd.VectorizeFloat64,
//		rosimd.ReduceSumFloat64,
//	)
func VectorizeFloat64[V Float64Buffer[V]](source ro.Observable[float64]) ro.Observable[V] {
	return vectorize[float64, V](source)
}

// ToScalarInt8 hands each vector's valid lanes downstream as a slice.
//
// It is the exit from vector space, the counterpart of VectorizeInt8. A short final
// batch yields a correspondingly short slice: padded lanes are never included, so the
// concatenation of every slice is exactly what went in.
//
// Its constraint asks only that a vector can report its lanes, so it accepts the
// standard library's vector types as well as this package's — including simd.Int64s and
// simd.Uint64s, which the arithmetic operators reject for want of Min and Max.
//
// It is not a curried operator: taking the source directly lets the type argument be
// inferred from the surrounding Pipe.
//
//	ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
//		source,
//		rosimd.VectorizeInt8,
//		rosimd.ToScalarInt8,
//		ro.Flatten[int8](),
//	)
//
// To go straight back to individual values, use FlattenInt8 instead.
func ToScalarInt8[V LaneStore[int8]](source ro.Observable[V]) ro.Observable[[]int8] {
	return emitLaneSlices[int8, V](source)
}

// ToScalarInt16 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarInt16[V LaneStore[int16]](source ro.Observable[V]) ro.Observable[[]int16] {
	return emitLaneSlices[int16, V](source)
}

// ToScalarInt32 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarInt32[V LaneStore[int32]](source ro.Observable[V]) ro.Observable[[]int32] {
	return emitLaneSlices[int32, V](source)
}

// ToScalarInt64 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarInt64[V LaneStore[int64]](source ro.Observable[V]) ro.Observable[[]int64] {
	return emitLaneSlices[int64, V](source)
}

// ToScalarUint8 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarUint8[V LaneStore[uint8]](source ro.Observable[V]) ro.Observable[[]uint8] {
	return emitLaneSlices[uint8, V](source)
}

// ToScalarUint16 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarUint16[V LaneStore[uint16]](source ro.Observable[V]) ro.Observable[[]uint16] {
	return emitLaneSlices[uint16, V](source)
}

// ToScalarUint32 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarUint32[V LaneStore[uint32]](source ro.Observable[V]) ro.Observable[[]uint32] {
	return emitLaneSlices[uint32, V](source)
}

// ToScalarUint64 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarUint64[V LaneStore[uint64]](source ro.Observable[V]) ro.Observable[[]uint64] {
	return emitLaneSlices[uint64, V](source)
}

// ToScalarFloat32 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarFloat32[V LaneStore[float32]](source ro.Observable[V]) ro.Observable[[]float32] {
	return emitLaneSlices[float32, V](source)
}

// ToScalarFloat64 hands each vector's valid lanes downstream as a slice. See ToScalarInt8.
func ToScalarFloat64[V LaneStore[float64]](source ro.Observable[V]) ro.Observable[[]float64] {
	return emitLaneSlices[float64, V](source)
}

// FlattenInt8 hands each vector's valid lanes downstream one at a time.
//
// It is ToScalarInt8 followed by ro.Flatten, in one stage: where ToScalarInt8 emits one
// slice per vector, this emits one value per lane, turning a vector stream back into the
// scalar stream VectorizeInt8 was given.
//
// Padded lanes are never emitted, so a stream that goes through VectorizeInt8 and back
// out through this arrives unchanged.
//
//	ro.Pipe2[int8, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8,
//		rosimd.FlattenInt8,
//	)
//
// Every lane of one vector carries that vector's context onward, so context propagation
// survives the round trip.
func FlattenInt8[V LaneStore[int8]](source ro.Observable[V]) ro.Observable[int8] {
	return emitLanes[int8, V](source)
}

// FlattenInt16 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenInt16[V LaneStore[int16]](source ro.Observable[V]) ro.Observable[int16] {
	return emitLanes[int16, V](source)
}

// FlattenInt32 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenInt32[V LaneStore[int32]](source ro.Observable[V]) ro.Observable[int32] {
	return emitLanes[int32, V](source)
}

// FlattenInt64 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenInt64[V LaneStore[int64]](source ro.Observable[V]) ro.Observable[int64] {
	return emitLanes[int64, V](source)
}

// FlattenUint8 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenUint8[V LaneStore[uint8]](source ro.Observable[V]) ro.Observable[uint8] {
	return emitLanes[uint8, V](source)
}

// FlattenUint16 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenUint16[V LaneStore[uint16]](source ro.Observable[V]) ro.Observable[uint16] {
	return emitLanes[uint16, V](source)
}

// FlattenUint32 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenUint32[V LaneStore[uint32]](source ro.Observable[V]) ro.Observable[uint32] {
	return emitLanes[uint32, V](source)
}

// FlattenUint64 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenUint64[V LaneStore[uint64]](source ro.Observable[V]) ro.Observable[uint64] {
	return emitLanes[uint64, V](source)
}

// FlattenFloat32 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenFloat32[V LaneStore[float32]](source ro.Observable[V]) ro.Observable[float32] {
	return emitLanes[float32, V](source)
}

// FlattenFloat64 hands each vector's valid lanes downstream one at a time. See FlattenInt8.
func FlattenFloat64[V LaneStore[float64]](source ro.Observable[V]) ro.Observable[float64] {
	return emitLanes[float64, V](source)
}
