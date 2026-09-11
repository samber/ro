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
	"simd"

	"github.com/samber/ro"
)

// Float32Vector describes the operations available on any float32 vector, whether it
// is the standard library's simd.Float32s or this package's PartialFloat32s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastFloat32s to
// widen one — doing so makes the function simd-dependent and breaks the compiler's
// own clone dispatcher. Callers widen scalars themselves with BroadcastFloat32.
//
// Div appears here and on no integer type: the standard library provides lane-wise
// division for the float types only.
type Float32Vector[V any] interface {
	LaneStore[float32]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Div(V) V
	Min(V) V
	Max(V) V
}

// Float32Buffer is a Float32Vector that can also be built from a slice of scalars.
// Only PartialFloat32s satisfies it.
type Float32Buffer[V any] interface {
	Float32Vector[V]
	LaneBuffer[V, float32]
}

// Float32Searchable is a Float32Vector that can test its own lanes for a value. Only
// PartialFloat32s satisfies it.
type Float32Searchable[V any] interface {
	Float32Vector[V]
	LaneMatcher[V]
}

// PartialFloat32s is a vector of float32 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the result,
// every operation applies mask so padded lanes keep whatever they held before — zero,
// as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialFloat32s struct {
	vec  simd.Float32s
	mask simd.Mask32s
	n    int
}

// BroadcastFloat32 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastFloat32(value float32) PartialFloat32s {
	vec := simd.BroadcastFloat32s(value)

	return PartialFloat32s{vec: vec, mask: fullMaskFloat32(), n: vec.Len()}
}

// fullMaskFloat32 is an all-true mask. simd.Float32s has no ToMask, so it is built
// through simd.Int32s — the resulting Mask32s is shared by both types of the 32-bit
// width. ToMask compares against zero, so broadcasting any non-zero value yields
// every lane set.
func fullMaskFloat32() simd.Mask32s { return simd.BroadcastInt32s(1).ToMask() }

// prefixMaskFloat32 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
//
// The ramp is built in the float domain rather than borrowed from simd.Int32s: lane
// indices are small integers, which float32 represents exactly, so the comparison is
// as precise as the integer one would be.
func prefixMaskFloat32(n int) simd.Mask32s {
	lanes := simd.BroadcastFloat32s(0).Len()

	var ramp [maxLanes]float32
	for i := range ramp {
		ramp[i] = float32(i)
	}

	return simd.LoadFloat32s(ramp[:lanes]).Less(simd.BroadcastFloat32s(float32(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialFloat32s) Load(src []float32) PartialFloat32s {
	vec := simd.LoadFloat32s(src)

	return PartialFloat32s{vec: vec, mask: fullMaskFloat32(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialFloat32s) LoadPart(src []float32, n int) PartialFloat32s {
	lanes := simd.BroadcastFloat32s(0).Len()

	var buf [maxLanes]float32
	copy(buf[:], src[:n])

	return PartialFloat32s{vec: simd.LoadFloat32s(buf[:lanes]), mask: prefixMaskFloat32(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialFloat32s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialFloat32s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialFloat32s) StorePart(dst []float32) int {
	var buf [maxLanes]float32
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialFloat32s) Values() []float32 {
	out := make([]float32, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
//
// It is also what keeps division safe: padded lanes hold zero, so an unmasked Div
// would turn them into NaN, and a later Min or Max would then propagate that NaN into
// a valid lane.
func (p PartialFloat32s) apply(result simd.Float32s) PartialFloat32s {
	return PartialFloat32s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialFloat32s) Add(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialFloat32s) Sub(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies other lane-wise, leaving padded lanes untouched.
func (p PartialFloat32s) Mul(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.vec.Mul(other.vec))
}

// Div divides by other lane-wise, leaving padded lanes untouched.
//
// Division by zero yields ±Inf and 0/0 yields NaN, exactly as Go's own float division
// does. Neither is an error, and neither is suppressed.
func (p PartialFloat32s) Div(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.vec.Div(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched. A lane
// where either operand is NaN yields NaN.
func (p PartialFloat32s) Min(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.restoreNaN(p.vec.Min(other.vec), other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched. A lane
// where either operand is NaN yields NaN.
func (p PartialFloat32s) Max(other PartialFloat32s) PartialFloat32s {
	return p.apply(p.restoreNaN(p.vec.Max(other.vec), other.vec))
}

// restoreNaN puts NaN back into every lane where either operand held one.
//
// Hardware disagrees about this: x86's MINPS/MAXPS discard NaN and return the other
// operand, while arm64's FMIN/FMAX propagate it. Rather than document an
// architecture-dependent result, detect the NaN lanes and force the answer, which
// costs two blends and makes every machine agree with Go's own min and max builtins.
//
// The test is v != v, true only for NaN under IEEE 754. IfElse keeps its receiver
// where the mask is set, so each line replaces the hardware's answer with the operand
// that was NaN.
func (p PartialFloat32s) restoreNaN(result, other simd.Float32s) simd.Float32s {
	result = p.vec.IfElse(p.vec.NotEqual(p.vec), result)

	return other.IfElse(other.NotEqual(other), result)
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects it
// at construction with a panic, this cannot detect it: the bounds are opaque vectors
// that a generic operator may not inspect. The composition is Max(lower) then
// Min(upper), so inverted bounds collapse every lane to upper.
//
// A NaN lane stays NaN, because Min and Max both propagate it. Core ro.Clamp agrees:
// its comparisons against a NaN are all false, so it returns the value unchanged.
func (p PartialFloat32s) Clamp(lower, upper PartialFloat32s) PartialFloat32s {
	return p.Max(lower).Min(upper)
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialFloat32s) Sum() float32 {
	var buf [maxLanes]float32
	n := p.StorePart(buf[:])

	total := float32(0)
	for i := range n {
		total += buf[i]
	}

	return total
}

// Contains reports, lane by lane, which valid lanes equal target.
//
// The result is a mask — SIMD's vector of booleans — already intersected with the
// validity mask, so padded lanes are never set. That intersection is what makes
// searching for zero correct: padding is zero-filled and would otherwise match.
//
// Searching for NaN never matches, since NaN compares unequal to everything including
// itself. That is what Go's own == does, so a scalar search would report the same.
//
// The comparison is lane-for-lane, not lane-against-every-lane. It answers "is this
// value present" only because BroadcastFloat32 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsFloat32
// to collapse a whole stream to a single bool.
func (p PartialFloat32s) Contains(target PartialFloat32s) simd.Mask32s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialFloat32s) Select(mask simd.Mask32s, other PartialFloat32s) PartialFloat32s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsFloat32 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
//
// The buffer is an integer one because a mask converts only to the integer vector type
// of its width; only the zero/non-zero distinction is read from it.
func (p PartialFloat32s) anyMatch(target PartialFloat32s) bool {
	var buf [maxLanes]int32
	p.Contains(target).ToInt32s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests can
// assert that padding is left untouched by element-wise operations.
func (p PartialFloat32s) rawLane(index int) float32 {
	var buf [maxLanes]float32
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
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

// SubFloat32 subtracts operand from every lane of every vector in the stream.
func SubFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// MulFloat32 multiplies every lane of every vector in the stream by operand.
func MulFloat32[V Float32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
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

// MinFloat32 keeps the smaller of each lane and the matching lane of ceiling.
//
// This is element-wise, one vector out per vector in — unlike ro.Min, which aggregates
// a whole stream into a single value. ReduceMinFloat32 is the aggregating counterpart,
// and the two treat NaN differently: see its documentation.
func MinFloat32[V Float32Vector[V]](ceiling V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Min(ceiling) })
	}
}

// MaxFloat32 keeps the larger of each lane and the matching lane of floor.
//
// This is element-wise. ReduceMaxFloat32 is the aggregating counterpart.
func MaxFloat32[V Float32Vector[V]](floor V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(floor) })
	}
}

// ClampFloat32 bounds every lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects it
// at construction with a panic, this cannot detect it: the bounds are opaque vectors
// that a generic operator may not inspect. The composition is Max(lower) then
// Min(upper), so inverted bounds collapse every lane to upper.
func ClampFloat32[V Float32Vector[V]](lower, upper V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(lower).Min(upper) })
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

// SubWithFloat32 subtracts another vector stream from this one, pairing vectors in order.
func SubWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// MulWithFloat32 multiplies this vector stream by another, pairing vectors in order.
func MulWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
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

// MinWithFloat32 keeps the smaller of each lane pair from two vector streams.
func MinWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Min(right) })
	}
}

// MaxWithFloat32 keeps the larger of each lane pair from two vector streams.
func MaxWithFloat32[V Float32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Max(right) })
	}
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
//		rosimd.VectorizeFloat32,
//		rosimd.ReduceSumFloat32,
//	)
func ReduceSumFloat32[V Float32Vector[V]](source ro.Observable[V]) ro.Observable[float32] {
	return reduceLanes(
		source,
		func(acc, lane float32) float32 { return acc + lane },
		func(acc float32, _ bool, emit func(float32)) { emit(acc) },
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

// ReduceContainsFloat32 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastFloat32 so every lane of target holds it:
//
//	rosimd.ReduceContainsFloat32(rosimd.BroadcastFloat32(4.2))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer. Searching for NaN always reports false,
// since NaN equals nothing, not even itself.
func ReduceContainsFloat32[V Float32Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}
