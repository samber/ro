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

import (
	"context"

	"github.com/samber/ro"
)

// Crossing the boundary of vector space, in both directions.
//
// Vectorize batches a scalar stream into vectors. ToScalar and Flatten bring it back out
// — the first as one slice per vector, the second as one value per lane. The Reduce
// operators in reduce.go are the other way out, collapsing a whole stream to one value.
//
// Vectorize produces the Partial types alone, since only they carry the validity mask a
// short final batch needs. The two exits are less demanding: they ask only that a vector
// can report its lanes, so the standard library's vector types work too.
//
// All three are curried, like the operators in core ro. That costs inference — a curried
// operator's type parameter appears only in the type of the func it returns, which Go
// does not reach — so each names its vector type at the call site.

// VectorizeInt8 batches a scalar stream into vectors.
//
// It emits a full vector every time the buffer fills, and on completion emits one final
// PartialInt8s holding whatever is left. Downstream operators see that short vector as a
// first-class value rather than a special case, because its padded lanes are masked out
// of every operation.
//
// The vector type is given at the call site, since currying puts it out of inference's
// reach:
//
//	ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
//		source,
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.ToScalar[rosimd.PartialInt8s](),
//		ro.Flatten[int8](),
//	)
//
// It produces the Partial types alone: the standard library's vector types expose no
// constructor method, and a generic function cannot reach the simd.LoadXxx package
// functions.
func VectorizeInt8[V Int8Buffer[V]]() func(ro.Observable[int8]) ro.Observable[V] {
	return func(source ro.Observable[int8]) ro.Observable[V] {
		return vectorize[int8, V](source)
	}
}

// VectorizeInt16 batches a scalar stream into vectors of int16 lanes. See VectorizeInt8.
func VectorizeInt16[V Int16Buffer[V]]() func(ro.Observable[int16]) ro.Observable[V] {
	return func(source ro.Observable[int16]) ro.Observable[V] {
		return vectorize[int16, V](source)
	}
}

// VectorizeInt32 batches a scalar stream into vectors of int32 lanes. See VectorizeInt8.
func VectorizeInt32[V Int32Buffer[V]]() func(ro.Observable[int32]) ro.Observable[V] {
	return func(source ro.Observable[int32]) ro.Observable[V] {
		return vectorize[int32, V](source)
	}
}

// VectorizeInt64 batches a scalar stream into vectors of int64 lanes. See VectorizeInt8.
func VectorizeInt64[V Int64Buffer[V]]() func(ro.Observable[int64]) ro.Observable[V] {
	return func(source ro.Observable[int64]) ro.Observable[V] {
		return vectorize[int64, V](source)
	}
}

// VectorizeUint8 batches a scalar stream into vectors of uint8 lanes. See VectorizeInt8.
func VectorizeUint8[V Uint8Buffer[V]]() func(ro.Observable[uint8]) ro.Observable[V] {
	return func(source ro.Observable[uint8]) ro.Observable[V] {
		return vectorize[uint8, V](source)
	}
}

// VectorizeUint16 batches a scalar stream into vectors of uint16 lanes. See VectorizeInt8.
func VectorizeUint16[V Uint16Buffer[V]]() func(ro.Observable[uint16]) ro.Observable[V] {
	return func(source ro.Observable[uint16]) ro.Observable[V] {
		return vectorize[uint16, V](source)
	}
}

// VectorizeUint32 batches a scalar stream into vectors of uint32 lanes. See VectorizeInt8.
func VectorizeUint32[V Uint32Buffer[V]]() func(ro.Observable[uint32]) ro.Observable[V] {
	return func(source ro.Observable[uint32]) ro.Observable[V] {
		return vectorize[uint32, V](source)
	}
}

// VectorizeUint64 batches a scalar stream into vectors of uint64 lanes. See VectorizeInt8.
func VectorizeUint64[V Uint64Buffer[V]]() func(ro.Observable[uint64]) ro.Observable[V] {
	return func(source ro.Observable[uint64]) ro.Observable[V] {
		return vectorize[uint64, V](source)
	}
}

// VectorizeFloat32 batches a scalar stream into vectors of float32 lanes. See VectorizeInt8.
func VectorizeFloat32[V Float32Buffer[V]]() func(ro.Observable[float32]) ro.Observable[V] {
	return func(source ro.Observable[float32]) ro.Observable[V] {
		return vectorize[float32, V](source)
	}
}

// VectorizeFloat64 batches a scalar stream into vectors of float64 lanes. See VectorizeInt8.
func VectorizeFloat64[V Float64Buffer[V]]() func(ro.Observable[float64]) ro.Observable[V] {
	return func(source ro.Observable[float64]) ro.Observable[V] {
		return vectorize[float64, V](source)
	}
}

// ToScalar hands each vector's valid lanes downstream as a slice.
//
// It is the exit from vector space, the counterpart of Vectorize. A short final batch
// yields a correspondingly short slice: padded lanes are never included, so the slices
// concatenated are exactly the stream that went in.
//
// One operator serves every element type. Only the vector type is named — the element
// type is read off that vector's own StorePart signature:
//
//	ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
//		source,
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.ToScalar[rosimd.PartialInt8s](),
//		ro.Flatten[int8](),
//	)
//
// Its constraint asks only that a vector can report its lanes, not that it can do
// arithmetic, so it accepts the standard library's vector types as well — simd.Int64s
// and simd.Uint64s included, which the arithmetic operators reject for want of Min and
// Max.
//
// To go straight back to individual values, use Flatten instead.
func ToScalar[V LaneStore[T], T any]() func(ro.Observable[V]) ro.Observable[[]T] {
	return func(source ro.Observable[V]) ro.Observable[[]T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[[]T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value V) {
						var buffer [maxLanes]T
						n := value.StorePart(buffer[:])

						lanes := make([]T, n)
						copy(lanes, buffer[:n])

						destination.NextWithContext(ctx, lanes)
					},
					destination.ErrorWithContext,
					destination.CompleteWithContext,
				),
			)

			return sub.Unsubscribe
		})
	}
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
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.Flatten[rosimd.PartialInt8s](),
//	)
//
// Like ToScalar, one operator serves every element type and only the vector type is
// named. Every lane of one vector carries that vector's context onward, so context
// propagation survives the round trip.
func Flatten[V LaneStore[T], T any]() func(ro.Observable[V]) ro.Observable[T] {
	return func(source ro.Observable[V]) ro.Observable[T] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value V) {
						var buffer [maxLanes]T
						n := value.StorePart(buffer[:])

						for i := range n {
							destination.NextWithContext(ctx, buffer[i])
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
