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

// Element-wise arithmetic: Add, Sub, Mul and Div, each paired with the two-stream With
// variant that combines vectors from a second stream in lockstep.
//
// Mul is absent for Int64 and Uint64, and Div exists only for the float types, because
// the standard library provides neither operation for those lanes.

// AddInt8 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastInt8, which also lets the type argument be inferred.
//
//	rosimd.AddInt8(rosimd.BroadcastInt8(42))   // stream of PartialInt8s
//	rosimd.AddInt8(simd.BroadcastInt8s(42))    // stream of simd.Int8s
func AddInt8[V Int8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddInt16 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastInt16, which also lets the type argument be inferred.
//
//	rosimd.AddInt16(rosimd.BroadcastInt16(42))   // stream of PartialInt16s
//	rosimd.AddInt16(simd.BroadcastInt16s(42))    // stream of simd.Int16s
func AddInt16[V Int16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddInt32 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastInt32, which also lets the type argument be inferred.
//
//	rosimd.AddInt32(rosimd.BroadcastInt32(42))   // stream of PartialInt32s
//	rosimd.AddInt32(simd.BroadcastInt32s(42))    // stream of simd.Int32s
func AddInt32[V Int32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddInt64 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastInt64, which also lets the type argument be inferred.
//
//	rosimd.AddInt64(rosimd.BroadcastInt64(42))   // stream of PartialInt64s
func AddInt64[V Int64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddUint8 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastUint8, which also lets the type argument be inferred.
//
//	rosimd.AddUint8(rosimd.BroadcastUint8(42))   // stream of PartialUint8s
//	rosimd.AddUint8(simd.BroadcastUint8s(42))    // stream of simd.Uint8s
func AddUint8[V Uint8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddUint16 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastUint16, which also lets the type argument be inferred.
//
//	rosimd.AddUint16(rosimd.BroadcastUint16(42))   // stream of PartialUint16s
//	rosimd.AddUint16(simd.BroadcastUint16s(42))    // stream of simd.Uint16s
func AddUint16[V Uint16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddUint32 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastUint32, which also lets the type argument be inferred.
//
//	rosimd.AddUint32(rosimd.BroadcastUint32(42))   // stream of PartialUint32s
//	rosimd.AddUint32(simd.BroadcastUint32s(42))    // stream of simd.Uint32s
func AddUint32[V Uint32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddUint64 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastUint64, which also lets the type argument be inferred.
//
//	rosimd.AddUint64(rosimd.BroadcastUint64(42))   // stream of PartialUint64s
func AddUint64[V Uint64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddFloat32 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastFloat32, which also lets the type argument be inferred.
//
//	rosimd.AddFloat32(rosimd.BroadcastFloat32(4.2))   // stream of PartialFloat32s
//	rosimd.AddFloat32(simd.BroadcastFloat32s(4.2))    // stream of simd.Float32s
func AddFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// AddFloat64 adds operand to every lane of every vector in the stream.
//
// operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic, and a
// generic operator cannot widen one for itself. Widen it at the call site with
// BroadcastFloat64, which also lets the type argument be inferred.
//
//	rosimd.AddFloat64(rosimd.BroadcastFloat64(4.2))   // stream of PartialFloat64s
//	rosimd.AddFloat64(simd.BroadcastFloat64s(4.2))    // stream of simd.Float64s
func AddFloat64[V Float64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Add(operand) })
	}
}

// SubInt8 subtracts operand from every lane of every vector in the stream.
func SubInt8[V Int8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubInt16 subtracts operand from every lane of every vector in the stream.
func SubInt16[V Int16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubInt32 subtracts operand from every lane of every vector in the stream.
func SubInt32[V Int32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubInt64 subtracts operand from every lane of every vector in the stream.
func SubInt64[V Int64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubUint8 subtracts operand from every lane of every vector in the stream.
//
// The subtraction wraps modulo 256, exactly as Go's uint8 arithmetic does; lanes
// smaller than the operand underflow to large values rather than saturating.
func SubUint8[V Uint8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubUint16 subtracts operand from every lane of every vector in the stream.
//
// The subtraction wraps modulo 65536, exactly as Go's uint16 arithmetic does; lanes
// smaller than the operand underflow to large values rather than saturating.
func SubUint16[V Uint16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubUint32 subtracts operand from every lane of every vector in the stream.
func SubUint32[V Uint32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubUint64 subtracts operand from every lane of every vector in the stream.
func SubUint64[V Uint64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubFloat32 subtracts operand from every lane of every vector in the stream.
func SubFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// SubFloat64 subtracts operand from every lane of every vector in the stream.
func SubFloat64[V Float64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// MulInt8 multiplies every lane of every vector in the stream by operand.
func MulInt8[V Int8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulInt16 multiplies every lane of every vector in the stream by operand.
func MulInt16[V Int16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulInt32 multiplies every lane of every vector in the stream by operand.
func MulInt32[V Int32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulUint8 multiplies every lane of every vector in the stream by operand.
func MulUint8[V Uint8Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulUint16 multiplies every lane of every vector in the stream by operand.
func MulUint16[V Uint16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulUint32 multiplies every lane of every vector in the stream by operand.
func MulUint32[V Uint32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulFloat32 multiplies every lane of every vector in the stream by operand.
func MulFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MulFloat64 multiplies every lane of every vector in the stream by operand.
func MulFloat64[V Float64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// DivFloat32 divides every lane of every vector in the stream by operand.
//
// A zero lane in operand yields ±Inf rather than an error, exactly as Go's own float
// division does.
func DivFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Div(operand) })
	}
}

// DivFloat64 divides every lane of every vector in the stream by operand.
//
// A zero lane in operand yields ±Inf rather than an error, exactly as Go's own float
// division does.
func DivFloat64[V Float64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Div(operand) })
	}
}

// AddWithInt8 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddInt8, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithInt8[V Int8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithInt16 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddInt16, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithInt32 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddInt32, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithInt64 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddInt64, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithInt64[V Int64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithUint8 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddUint8, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithUint8[V Uint8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithUint16 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddUint16, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithUint16[V Uint16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithUint32 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddUint32, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithUint32[V Uint32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithUint64 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddUint64, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithUint64[V Uint64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithFloat32 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddFloat32, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// AddWithFloat64 adds another vector stream to this one, pairing vectors in order.
//
// It is the curried, two-stream counterpart of AddFloat64, following the same naming
// convention as ro.ZipWith and ro.MergeWith.
func AddWithFloat64[V Float64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Add(right) })
	}
}

// SubWithInt8 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt8[V Int8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithInt16 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithInt32 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithInt64 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt64[V Int64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithUint8 subtracts another vector stream from this one, pairing vectors in order.
func SubWithUint8[V Uint8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithUint16 subtracts another vector stream from this one, pairing vectors in order.
func SubWithUint16[V Uint16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithUint32 subtracts another vector stream from this one, pairing vectors in order.
func SubWithUint32[V Uint32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithUint64 subtracts another vector stream from this one, pairing vectors in order.
func SubWithUint64[V Uint64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithFloat32 subtracts another vector stream from this one, pairing vectors in order.
func SubWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// SubWithFloat64 subtracts another vector stream from this one, pairing vectors in order.
func SubWithFloat64[V Float64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// MulWithInt8 multiplies this vector stream by another, pairing vectors in order.
func MulWithInt8[V Int8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithInt16 multiplies this vector stream by another, pairing vectors in order.
func MulWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithInt32 multiplies this vector stream by another, pairing vectors in order.
func MulWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithUint8 multiplies this vector stream by another, pairing vectors in order.
func MulWithUint8[V Uint8Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithUint16 multiplies this vector stream by another, pairing vectors in order.
func MulWithUint16[V Uint16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithUint32 multiplies this vector stream by another, pairing vectors in order.
func MulWithUint32[V Uint32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithFloat32 multiplies this vector stream by another, pairing vectors in order.
func MulWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MulWithFloat64 multiplies this vector stream by another, pairing vectors in order.
func MulWithFloat64[V Float64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// DivWithFloat32 divides this vector stream by another, pairing vectors in order.
func DivWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Div(right) })
	}
}

// DivWithFloat64 divides this vector stream by another, pairing vectors in order.
func DivWithFloat64[V Float64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Div(right) })
	}
}
