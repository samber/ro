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

// Int64Vector describes the operations available on an int64 vector.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastInt64s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastInt64.
//
// Unlike Int8Vector there is no Mul: the standard library provides no 64-bit lane
// multiply, so no int64 vector type can offer one.
//
// simd.Int64s does not satisfy this interface either — it has no Min or Max methods —
// so in practice only PartialInt64s, which synthesizes both from Less and IfElse,
// flows through the operators.
type Int64Vector[V any] interface {
	LaneStore[int64]

	Add(V) V
	Sub(V) V
	Min(V) V
	Max(V) V
}

// Int64Buffer is an Int64Vector that can also be built from a slice of scalars. Only
// PartialInt64s satisfies it.
type Int64Buffer[V any] interface {
	Int64Vector[V]
	LaneBuffer[V, int64]
}

// Int64Searchable is an Int64Vector that can test its own lanes for a value. Only
// PartialInt64s satisfies it.
type Int64Searchable[V any] interface {
	Int64Vector[V]
	LaneMatcher[V]
}

// PartialInt64s is a vector of int64 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialInt64s struct {
	vec  simd.Int64s
	mask simd.Mask64s
	n    int
}

// BroadcastInt64 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastInt64(value int64) PartialInt64s {
	vec := simd.BroadcastInt64s(value)

	return PartialInt64s{vec: vec, mask: fullMaskInt64(), n: vec.Len()}
}

// fullMaskInt64 is an all-true mask. ToMask compares against zero, so broadcasting
// any non-zero value yields every lane set.
func fullMaskInt64() simd.Mask64s { return simd.BroadcastInt64s(1).ToMask() }

// prefixMaskInt64 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
func prefixMaskInt64(n int) simd.Mask64s {
	lanes := simd.BroadcastInt64s(0).Len()

	var ramp [maxLanes]int64
	for i := range ramp {
		ramp[i] = int64(i)
	}

	return simd.LoadInt64s(ramp[:lanes]).Less(simd.BroadcastInt64s(int64(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialInt64s) Load(src []int64) PartialInt64s {
	vec := simd.LoadInt64s(src)

	return PartialInt64s{vec: vec, mask: fullMaskInt64(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialInt64s) LoadPart(src []int64, n int) PartialInt64s {
	lanes := simd.BroadcastInt64s(0).Len()

	var buf [maxLanes]int64
	copy(buf[:], src[:n])

	return PartialInt64s{vec: simd.LoadInt64s(buf[:lanes]), mask: prefixMaskInt64(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialInt64s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialInt64s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialInt64s) StorePart(dst []int64) int {
	var buf [maxLanes]int64
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialInt64s) Values() []int64 {
	out := make([]int64, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialInt64s) apply(result simd.Int64s) PartialInt64s {
	return PartialInt64s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialInt64s) Add(other PartialInt64s) PartialInt64s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialInt64s) Sub(other PartialInt64s) PartialInt64s {
	return p.apply(p.vec.Sub(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
//
// simd.Int64s has no Min method, so it is synthesized from Less and IfElse:
// x.IfElse(mask, y) keeps x where mask is set, so selecting on x.Less(y) keeps x
// exactly in the lanes where it is the smaller.
func (p PartialInt64s) Min(other PartialInt64s) PartialInt64s {
	return p.apply(p.vec.IfElse(p.vec.Less(other.vec), other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
//
// Synthesized like Min, with the IfElse operands swapped: where p is less than
// other, keep other.
func (p PartialInt64s) Max(other PartialInt64s) PartialInt64s {
	return p.apply(other.vec.IfElse(p.vec.Less(other.vec), p.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
//
// It composes the synthesized Max and Min rather than repeating their IfElse
// selections, so the operand order is stated in exactly one place. The double mask
// application is a no-op: Max has already restored the padded lanes.
func (p PartialInt64s) Clamp(lower, upper PartialInt64s) PartialInt64s {
	return p.Max(lower).Min(upper)
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialInt64s) Sum() int64 {
	var buf [maxLanes]int64
	n := p.StorePart(buf[:])

	total := int64(0)
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
// value present" only because BroadcastInt64 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsInt64
// to collapse a whole stream to a single bool.
func (p PartialInt64s) Contains(target PartialInt64s) simd.Mask64s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialInt64s) Select(mask simd.Mask64s, other PartialInt64s) PartialInt64s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsInt64 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialInt64s) anyMatch(target PartialInt64s) bool {
	var buf [maxLanes]int64
	p.Contains(target).ToInt64s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests
// can assert that padding is left untouched by element-wise operations.
func (p PartialInt64s) rawLane(index int) int64 {
	var buf [maxLanes]int64
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
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

// SubInt64 subtracts operand from every lane of every vector in the stream.
func SubInt64[V Int64Vector[V]](operand V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Sub(operand) })
	}
}

// MinInt64 keeps the smaller of each lane and the matching lane of ceiling.
//
// This is element-wise, one vector out per vector in — unlike ro.Min, which
// aggregates a whole stream into a single value. ReduceMinInt64 is the aggregating
// counterpart.
func MinInt64[V Int64Vector[V]](ceiling V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Min(ceiling) })
	}
}

// MaxInt64 keeps the larger of each lane and the matching lane of floor.
//
// This is element-wise. ReduceMaxInt64 is the aggregating counterpart.
func MaxInt64[V Int64Vector[V]](floor V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(floor) })
	}
}

// ClampInt64 bounds every lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func ClampInt64[V Int64Vector[V]](lower, upper V) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return mapVector(source, func(value V) V { return value.Max(lower).Min(upper) })
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

// SubWithInt64 subtracts another vector stream from this one, pairing vectors in order.
func SubWithInt64[V Int64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Sub(right) })
	}
}

// MinWithInt64 keeps the smaller of each lane pair from two vector streams.
func MinWithInt64[V Int64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Min(right) })
	}
}

// MaxWithInt64 keeps the larger of each lane pair from two vector streams.
func MaxWithInt64[V Int64Vector[V]](other ro.Observable[V]) func(ro.Observable[V]) ro.Observable[V] {
	return func(source ro.Observable[V]) ro.Observable[V] {
		return zipVector(source, other, func(left, right V) V { return left.Max(right) })
	}
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
//		rosimd.VectorizeInt64,
//		rosimd.ReduceSumInt64,
//	)
func ReduceSumInt64[V Int64Vector[V]](source ro.Observable[V]) ro.Observable[int64] {
	return reduceLanes(
		source,
		func(acc, lane int64) int64 { return acc + lane },
		func(acc int64, _ bool, emit func(int64)) { emit(acc) },
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

// ReduceContainsInt64 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt64 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt64(rosimd.BroadcastInt64(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt64[V Int64Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}
