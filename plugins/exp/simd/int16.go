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

// Int16Vector describes the operations available on any int16 vector, whether it is
// the standard library's simd.Int16s or this package's PartialInt16s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastInt16s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastInt16.
type Int16Vector[V any] interface {
	LaneStore[int16]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Min(V) V
	Max(V) V
}

// Int16Buffer is an Int16Vector that can also be built from a slice of scalars. Only
// PartialInt16s satisfies it.
type Int16Buffer[V any] interface {
	Int16Vector[V]
	LaneBuffer[V, int16]
}

// Int16Searchable is an Int16Vector that can test its own lanes for a value. Only
// PartialInt16s satisfies it.
type Int16Searchable[V any] interface {
	Int16Vector[V]
	LaneMatcher[V]
}

// PartialInt16s is a vector of int16 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialInt16s struct {
	vec  simd.Int16s
	mask simd.Mask16s
	n    int
}

// BroadcastInt16 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastInt16(value int16) PartialInt16s {
	vec := simd.BroadcastInt16s(value)

	return PartialInt16s{vec: vec, mask: fullMaskInt16(), n: vec.Len()}
}

// fullMaskInt16 is an all-true mask. ToMask compares against zero, so broadcasting
// any non-zero value yields every lane set.
func fullMaskInt16() simd.Mask16s { return simd.BroadcastInt16s(1).ToMask() }

// prefixMaskInt16 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
func prefixMaskInt16(n int) simd.Mask16s {
	lanes := simd.BroadcastInt16s(0).Len()

	var ramp [maxLanes]int16
	for i := range ramp {
		ramp[i] = int16(i)
	}

	return simd.LoadInt16s(ramp[:lanes]).Less(simd.BroadcastInt16s(int16(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialInt16s) Load(src []int16) PartialInt16s {
	vec := simd.LoadInt16s(src)

	return PartialInt16s{vec: vec, mask: fullMaskInt16(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialInt16s) LoadPart(src []int16, n int) PartialInt16s {
	lanes := simd.BroadcastInt16s(0).Len()

	var buf [maxLanes]int16
	copy(buf[:], src[:n])

	return PartialInt16s{vec: simd.LoadInt16s(buf[:lanes]), mask: prefixMaskInt16(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialInt16s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialInt16s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialInt16s) StorePart(dst []int16) int {
	var buf [maxLanes]int16
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialInt16s) Values() []int16 {
	out := make([]int16, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialInt16s) apply(result simd.Int16s) PartialInt16s {
	return PartialInt16s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialInt16s) Add(other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialInt16s) Sub(other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies by other lane-wise, leaving padded lanes untouched.
func (p PartialInt16s) Mul(other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Mul(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
func (p PartialInt16s) Min(other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Min(other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
func (p PartialInt16s) Max(other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Max(other.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func (p PartialInt16s) Clamp(lower, upper PartialInt16s) PartialInt16s {
	return p.apply(p.vec.Max(lower.vec).Min(upper.vec))
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialInt16s) Sum() int16 {
	var buf [maxLanes]int16
	n := p.StorePart(buf[:])

	total := int16(0)
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
// value present" only because BroadcastInt16 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsInt16
// to collapse a whole stream to a single bool.
func (p PartialInt16s) Contains(target PartialInt16s) simd.Mask16s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialInt16s) Select(mask simd.Mask16s, other PartialInt16s) PartialInt16s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsInt16 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialInt16s) anyMatch(target PartialInt16s) bool {
	var buf [maxLanes]int16
	p.Contains(target).ToInt16s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests
// can assert that padding is left untouched by element-wise operations.
func (p PartialInt16s) rawLane(index int) int16 {
	var buf [maxLanes]int16
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
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

// SubInt16 subtracts operand from every lane of every vector in the stream.
func SubInt16[V Int16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// MulInt16 multiplies every lane of every vector in the stream by operand.
func MulInt16[V Int16Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Mul(operand) })
	}
}

// MinInt16 keeps the smaller of each lane and the matching lane of ceiling.
//
// This is element-wise, one vector out per vector in — unlike ro.Min, which
// aggregates a whole stream into a single value. ReduceMinInt16 is the aggregating
// counterpart.
func MinInt16[V Int16Vector[V]](ceiling V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Min(ceiling) })
	}
}

// MaxInt16 keeps the larger of each lane and the matching lane of floor.
//
// This is element-wise. ReduceMaxInt16 is the aggregating counterpart.
func MaxInt16[V Int16Vector[V]](floor V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(floor) })
	}
}

// ClampInt16 bounds every lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func ClampInt16[V Int16Vector[V]](lower, upper V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(lower).Min(upper) })
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

// SubWithInt16 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// MulWithInt16 multiplies this vector stream by another, pairing vectors in order.
func MulWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Mul(right) })
	}
}

// MinWithInt16 keeps the smaller of each lane pair from two vector streams.
func MinWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Min(right) })
	}
}

// MaxWithInt16 keeps the larger of each lane pair from two vector streams.
func MaxWithInt16[V Int16Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Max(right) })
	}
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
//		rosimd.VectorizeInt16,
//		rosimd.ReduceSumInt16,
//	)
func ReduceSumInt16[V Int16Vector[V]](source ro.Observable[V]) ro.Observable[int16] {
	return reduceLanes(
		source,
		func(acc, lane int16) int16 { return acc + lane },
		func(acc int16, _ bool, emit func(int16)) { emit(acc) },
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

// ReduceContainsInt16 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt16 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt16(rosimd.BroadcastInt16(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt16[V Int16Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}
