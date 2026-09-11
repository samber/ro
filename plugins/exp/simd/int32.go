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

// Int32Vector describes the operations available on any int32 vector, whether it is
// the standard library's simd.Int32s or this package's PartialInt32s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastInt32s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastInt32.
type Int32Vector[V any] interface {
	LaneStore[int32]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Min(V) V
	Max(V) V
}

// Int32Buffer is an Int32Vector that can also be built from a slice of scalars. Only
// PartialInt32s satisfies it.
type Int32Buffer[V any] interface {
	Int32Vector[V]
	LaneBuffer[V, int32]
}

// Int32Searchable is an Int32Vector that can test its own lanes for a value. Only
// PartialInt32s satisfies it.
type Int32Searchable[V any] interface {
	Int32Vector[V]
	LaneMatcher[V]
}

// PartialInt32s is a vector of int32 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialInt32s struct {
	vec  simd.Int32s
	mask simd.Mask32s
	n    int
}

// BroadcastInt32 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastInt32(value int32) PartialInt32s {
	vec := simd.BroadcastInt32s(value)

	return PartialInt32s{vec: vec, mask: fullMaskInt32(), n: vec.Len()}
}

// fullMaskInt32 is an all-true mask. ToMask compares against zero, so broadcasting
// any non-zero value yields every lane set.
func fullMaskInt32() simd.Mask32s { return simd.BroadcastInt32s(1).ToMask() }

// prefixMaskInt32 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
func prefixMaskInt32(n int) simd.Mask32s {
	lanes := simd.BroadcastInt32s(0).Len()

	var ramp [maxLanes]int32
	for i := range ramp {
		ramp[i] = int32(i)
	}

	return simd.LoadInt32s(ramp[:lanes]).Less(simd.BroadcastInt32s(int32(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialInt32s) Load(src []int32) PartialInt32s {
	vec := simd.LoadInt32s(src)

	return PartialInt32s{vec: vec, mask: fullMaskInt32(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialInt32s) LoadPart(src []int32, n int) PartialInt32s {
	lanes := simd.BroadcastInt32s(0).Len()

	var buf [maxLanes]int32
	copy(buf[:], src[:n])

	return PartialInt32s{vec: simd.LoadInt32s(buf[:lanes]), mask: prefixMaskInt32(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialInt32s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialInt32s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialInt32s) StorePart(dst []int32) int {
	var buf [maxLanes]int32
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialInt32s) Values() []int32 {
	out := make([]int32, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialInt32s) apply(result simd.Int32s) PartialInt32s {
	return PartialInt32s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialInt32s) Add(other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialInt32s) Sub(other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies by other lane-wise, leaving padded lanes untouched.
func (p PartialInt32s) Mul(other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Mul(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
func (p PartialInt32s) Min(other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Min(other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
func (p PartialInt32s) Max(other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Max(other.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func (p PartialInt32s) Clamp(lower, upper PartialInt32s) PartialInt32s {
	return p.apply(p.vec.Max(lower.vec).Min(upper.vec))
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialInt32s) Sum() int32 {
	var buf [maxLanes]int32
	n := p.StorePart(buf[:])

	total := int32(0)
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
// The comparison is lane-for-lane, not lane-against-every-lane. It answers "is this
// value present" only because BroadcastInt32 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsInt32
// to collapse a whole stream to a single bool.
func (p PartialInt32s) Contains(target PartialInt32s) simd.Mask32s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialInt32s) Select(mask simd.Mask32s, other PartialInt32s) PartialInt32s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsInt32 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialInt32s) anyMatch(target PartialInt32s) bool {
	var buf [maxLanes]int32
	p.Contains(target).ToInt32s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests
// can assert that padding is left untouched by element-wise operations.
func (p PartialInt32s) rawLane(index int) int32 {
	var buf [maxLanes]int32
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
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

// SubInt32 subtracts operand from every lane of every vector in the stream.
func SubInt32[V Int32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// MulInt32 multiplies every lane of every vector in the stream by operand.
func MulInt32[V Int32Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MinInt32 keeps the smaller of each lane and the matching lane of ceiling.
//
// This is element-wise, one vector out per vector in — unlike ro.Min, which
// aggregates a whole stream into a single value. ReduceMinInt32 is the aggregating
// counterpart.
func MinInt32[V Int32Vector[V]](ceiling V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Min(ceiling) })
	}
}

// MaxInt32 keeps the larger of each lane and the matching lane of floor.
//
// This is element-wise. ReduceMaxInt32 is the aggregating counterpart.
func MaxInt32[V Int32Vector[V]](floor V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(floor) })
	}
}

// ClampInt32 bounds every lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func ClampInt32[V Int32Vector[V]](lower, upper V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(lower).Min(upper) })
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

// SubWithInt32 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// MulWithInt32 multiplies this vector stream by another, pairing vectors in order.
func MulWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MinWithInt32 keeps the smaller of each lane pair from two vector streams.
func MinWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Min(right) })
	}
}

// MaxWithInt32 keeps the larger of each lane pair from two vector streams.
func MaxWithInt32[V Int32Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Max(right) })
	}
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
//		rosimd.VectorizeInt32,
//		rosimd.ReduceSumInt32,
//	)
func ReduceSumInt32[V Int32Vector[V]](source ro.Observable[V]) ro.Observable[int32] {
	return reduceLanes(
		source,
		func(acc, lane int32) int32 { return acc + lane },
		func(acc int32, _ bool, emit func(int32)) { emit(acc) },
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

// ReduceContainsInt32 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt32 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt32(rosimd.BroadcastInt32(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt32[V Int32Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}
