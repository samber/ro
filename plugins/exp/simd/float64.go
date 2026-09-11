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

import "simd"

// Float64Vector describes the operations available on any float64 vector, whether it
// is the standard library's simd.Float64s or this package's PartialFloat64s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastFloat64s to
// widen one — doing so makes the function simd-dependent and breaks the compiler's
// own clone dispatcher. Callers widen scalars themselves with BroadcastFloat64.
//
// Div appears here and on no integer type: the standard library provides lane-wise
// division for the float types only.
type Float64Vector[V any] interface {
	LaneStore[float64]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Div(V) V
	Min(V) V
	Max(V) V
}

// Float64Buffer is a Float64Vector that can also be built from a slice of scalars.
// Only PartialFloat64s satisfies it.
type Float64Buffer[V any] interface {
	Float64Vector[V]
	LaneBuffer[V, float64]
}

// Float64Searchable is a Float64Vector that can test its own lanes for a value. Only
// PartialFloat64s satisfies it.
type Float64Searchable[V any] interface {
	Float64Vector[V]
	LaneMatcher[V]
}

// PartialFloat64s is a vector of float64 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the result,
// every operation applies mask so padded lanes keep whatever they held before — zero,
// as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialFloat64s struct {
	vec  simd.Float64s
	mask simd.Mask64s
	n    int
}

// BroadcastFloat64 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastFloat64(value float64) PartialFloat64s {
	vec := simd.BroadcastFloat64s(value)

	return PartialFloat64s{vec: vec, mask: fullMaskFloat64(), n: vec.Len()}
}

// fullMaskFloat64 is an all-true mask. simd.Float64s has no ToMask, so it is built
// through simd.Int64s — the resulting Mask64s is shared by both types of the 64-bit
// width. ToMask compares against zero, so broadcasting any non-zero value yields
// every lane set.
func fullMaskFloat64() simd.Mask64s { return simd.BroadcastInt64s(1).ToMask() }

// prefixMaskFloat64 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
//
// The ramp is built in the float domain rather than borrowed from simd.Int64s: lane
// indices are small integers, which float64 represents exactly, so the comparison is
// as precise as the integer one would be.
func prefixMaskFloat64(n int) simd.Mask64s {
	lanes := simd.BroadcastFloat64s(0).Len()

	var ramp [maxLanes]float64
	for i := range ramp {
		ramp[i] = float64(i)
	}

	return simd.LoadFloat64s(ramp[:lanes]).Less(simd.BroadcastFloat64s(float64(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialFloat64s) Load(src []float64) PartialFloat64s {
	vec := simd.LoadFloat64s(src)

	return PartialFloat64s{vec: vec, mask: fullMaskFloat64(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialFloat64s) LoadPart(src []float64, n int) PartialFloat64s {
	lanes := simd.BroadcastFloat64s(0).Len()

	var buf [maxLanes]float64
	copy(buf[:], src[:n])

	return PartialFloat64s{vec: simd.LoadFloat64s(buf[:lanes]), mask: prefixMaskFloat64(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialFloat64s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialFloat64s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialFloat64s) StorePart(dst []float64) int {
	var buf [maxLanes]float64
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialFloat64s) Values() []float64 {
	out := make([]float64, p.n)
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
func (p PartialFloat64s) apply(result simd.Float64s) PartialFloat64s {
	return PartialFloat64s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialFloat64s) Add(other PartialFloat64s) PartialFloat64s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialFloat64s) Sub(other PartialFloat64s) PartialFloat64s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies other lane-wise, leaving padded lanes untouched.
func (p PartialFloat64s) Mul(other PartialFloat64s) PartialFloat64s {
	return p.apply(p.vec.Mul(other.vec))
}

// Div divides by other lane-wise, leaving padded lanes untouched.
//
// Division by zero yields ±Inf and 0/0 yields NaN, exactly as Go's own float division
// does. Neither is an error, and neither is suppressed.
func (p PartialFloat64s) Div(other PartialFloat64s) PartialFloat64s {
	return p.apply(p.vec.Div(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched. A lane
// where either operand is NaN yields NaN.
func (p PartialFloat64s) Min(other PartialFloat64s) PartialFloat64s {
	return p.apply(p.restoreNaN(p.vec.Min(other.vec), other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched. A lane
// where either operand is NaN yields NaN.
func (p PartialFloat64s) Max(other PartialFloat64s) PartialFloat64s {
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
func (p PartialFloat64s) restoreNaN(result, other simd.Float64s) simd.Float64s {
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
func (p PartialFloat64s) Clamp(lower, upper PartialFloat64s) PartialFloat64s {
	return p.Max(lower).Min(upper)
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialFloat64s) Sum() float64 {
	var buf [maxLanes]float64
	n := p.StorePart(buf[:])

	total := float64(0)
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
// value present" only because BroadcastFloat64 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsFloat64
// to collapse a whole stream to a single bool.
func (p PartialFloat64s) Contains(target PartialFloat64s) simd.Mask64s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialFloat64s) Select(mask simd.Mask64s, other PartialFloat64s) PartialFloat64s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsFloat64 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
//
// The buffer is an integer one because a mask converts only to the integer vector type
// of its width; only the zero/non-zero distinction is read from it.
func (p PartialFloat64s) anyMatch(target PartialFloat64s) bool {
	var buf [maxLanes]int64
	p.Contains(target).ToInt64s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests can
// assert that padding is left untouched by element-wise operations.
func (p PartialFloat64s) rawLane(index int) float64 {
	var buf [maxLanes]float64
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
}
