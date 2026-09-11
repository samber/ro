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

// ToScalar hands each vector's valid lanes downstream as a slice.
//
// It is the exit from vector space, the counterpart of Vectorize. A short final batch
// yields a correspondingly short slice: padded lanes are never included, so the slices
// concatenated are exactly the stream that went in.
//
// One operator serves every element type. The element type is read off the vector's own
// StorePart signature, so both type arguments are inferred and the call site names
// neither:
//
//	ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
//		source,
//		rosimd.VectorizeInt8,
//		rosimd.ToScalar,
//		ro.Flatten[int8](),
//	)
//
// Its constraint asks only that a vector can report its lanes, not that it can do
// arithmetic, so it accepts the standard library's vector types as well — simd.Int64s
// and simd.Uint64s included, which the arithmetic operators reject for want of Min and
// Max.
//
// To go straight back to individual values, use Flatten instead.
func ToScalar[T any, V LaneStore[T]](source ro.Observable[V]) ro.Observable[[]T] {
	return emitLaneSlices[T, V](source)
}

// Flatten hands each vector's valid lanes downstream one at a time.
//
// It is ToScalar followed by ro.Flatten, in one stage: where ToScalar emits one slice per
// vector, this emits one value per lane, turning a vector stream back into the scalar
// stream Vectorize was given.
//
// Padded lanes are never emitted, so a stream that goes through Vectorize and back out
// through this arrives unchanged.
//
//	ro.Pipe2[int8, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8,
//		rosimd.Flatten,
//	)
//
// Like ToScalar, one operator serves every element type and both type arguments are
// inferred. Every lane of one vector carries that vector's context onward, so context
// propagation survives the round trip.
func Flatten[T any, V LaneStore[T]](source ro.Observable[V]) ro.Observable[T] {
	return emitLanes[T, V](source)
}
