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
