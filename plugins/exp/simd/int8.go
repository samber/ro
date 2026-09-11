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

// Int8Vector describes the operations available on any int8 vector, whether it is
// the standard library's simd.Int8s or this package's PartialInt8s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastInt8s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastInt8.
type Int8Vector[V any] interface {
	LaneStore[int8]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Min(V) V
	Max(V) V
}

// Int8Buffer is an Int8Vector that can also be built from a slice of scalars. Only
// PartialInt8s satisfies it.
type Int8Buffer[V any] interface {
	Int8Vector[V]
	LaneBuffer[V, int8]
}

// Int8Searchable is an Int8Vector that can test its own lanes for a value. Only
// PartialInt8s satisfies it.
type Int8Searchable[V any] interface {
	Int8Vector[V]
	LaneMatcher[V]
}

// PartialInt8s is a vector of int8 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialInt8s struct {
	vec  simd.Int8s
	mask simd.Mask8s
	n    int
}

// BroadcastInt8 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastInt8(value int8) PartialInt8s {
	vec := simd.BroadcastInt8s(value)

	return PartialInt8s{vec: vec, mask: fullMaskInt8(), n: vec.Len()}
}

// fullMaskInt8 is an all-true mask. ToMask compares against zero, so broadcasting
// any non-zero value yields every lane set.
func fullMaskInt8() simd.Mask8s { return simd.BroadcastInt8s(1).ToMask() }

// prefixMaskInt8 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
func prefixMaskInt8(n int) simd.Mask8s {
	lanes := simd.BroadcastInt8s(0).Len()

	var ramp [maxLanes]int8
	for i := range ramp {
		ramp[i] = int8(i)
	}

	return simd.LoadInt8s(ramp[:lanes]).Less(simd.BroadcastInt8s(int8(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialInt8s) Load(src []int8) PartialInt8s {
	vec := simd.LoadInt8s(src)

	return PartialInt8s{vec: vec, mask: fullMaskInt8(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialInt8s) LoadPart(src []int8, n int) PartialInt8s {
	lanes := simd.BroadcastInt8s(0).Len()

	var buf [maxLanes]int8
	copy(buf[:], src[:n])

	return PartialInt8s{vec: simd.LoadInt8s(buf[:lanes]), mask: prefixMaskInt8(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialInt8s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialInt8s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialInt8s) StorePart(dst []int8) int {
	var buf [maxLanes]int8
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialInt8s) Values() []int8 {
	out := make([]int8, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialInt8s) apply(result simd.Int8s) PartialInt8s {
	return PartialInt8s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialInt8s) Add(other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialInt8s) Sub(other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies by other lane-wise, leaving padded lanes untouched.
func (p PartialInt8s) Mul(other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Mul(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
func (p PartialInt8s) Min(other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Min(other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
func (p PartialInt8s) Max(other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Max(other.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func (p PartialInt8s) Clamp(lower, upper PartialInt8s) PartialInt8s {
	return p.apply(p.vec.Max(lower.vec).Min(upper.vec))
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialInt8s) Sum() int8 {
	var buf [maxLanes]int8
	n := p.StorePart(buf[:])

	total := int8(0)
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
// value present" only because BroadcastInt8 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsInt8
// to collapse a whole stream to a single bool.
func (p PartialInt8s) Contains(target PartialInt8s) simd.Mask8s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialInt8s) Select(mask simd.Mask8s, other PartialInt8s) PartialInt8s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsInt8 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialInt8s) anyMatch(target PartialInt8s) bool {
	var buf [maxLanes]int8
	p.Contains(target).ToInt8s().Store(buf[:p.vec.Len()])

	for _, lane := range buf[:p.vec.Len()] {
		if lane != 0 {
			return true
		}
	}

	return false
}

// rawLane exposes a lane of the underlying vector regardless of validity, so tests
// can assert that padding is left untouched by element-wise operations.
func (p PartialInt8s) rawLane(index int) int8 {
	var buf [maxLanes]int8
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
}
