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

// Uint16Vector describes the operations available on any uint16 vector, whether it is
// the standard library's simd.Uint16s or this package's PartialUint16s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastUint16s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastUint16.
type Uint16Vector[V any] interface {
	LaneStore[uint16]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Min(V) V
	Max(V) V
}

// Uint16Buffer is a Uint16Vector that can also be built from a slice of scalars. Only
// PartialUint16s satisfies it.
type Uint16Buffer[V any] interface {
	Uint16Vector[V]
	LaneBuffer[V, uint16]
}

// Uint16Searchable is a Uint16Vector that can test its own lanes for a value. Only
// PartialUint16s satisfies it.
type Uint16Searchable[V any] interface {
	Uint16Vector[V]
	LaneMatcher[V]
}

// PartialUint16s is a vector of uint16 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialUint16s struct {
	vec  simd.Uint16s
	mask simd.Mask16s
	n    int
}

// BroadcastUint16 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastUint16(value uint16) PartialUint16s {
	vec := simd.BroadcastUint16s(value)

	return PartialUint16s{vec: vec, mask: fullMaskUint16(), n: vec.Len()}
}

// fullMaskUint16 is an all-true mask. simd.Uint16s has no ToMask, but masks are per
// bit-width and therefore interchangeable between the signed and unsigned type of
// that width, so build it through simd.Int16s. ToMask compares against zero, so
// broadcasting any non-zero value yields every lane set.
func fullMaskUint16() simd.Mask16s { return simd.BroadcastInt16s(1).ToMask() }

// prefixMaskUint16 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp against n through simd.Int16s —
// the resulting Mask16s is shared by both types of the 16-bit width.
func prefixMaskUint16(n int) simd.Mask16s {
	lanes := simd.BroadcastUint16s(0).Len()

	var ramp [maxLanes]int16
	for i := range ramp {
		ramp[i] = int16(i)
	}

	return simd.LoadInt16s(ramp[:lanes]).Less(simd.BroadcastInt16s(int16(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialUint16s) Load(src []uint16) PartialUint16s {
	vec := simd.LoadUint16s(src)

	return PartialUint16s{vec: vec, mask: fullMaskUint16(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialUint16s) LoadPart(src []uint16, n int) PartialUint16s {
	lanes := simd.BroadcastUint16s(0).Len()

	var buf [maxLanes]uint16
	copy(buf[:], src[:n])

	return PartialUint16s{vec: simd.LoadUint16s(buf[:lanes]), mask: prefixMaskUint16(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialUint16s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialUint16s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialUint16s) StorePart(dst []uint16) int {
	var buf [maxLanes]uint16
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialUint16s) Values() []uint16 {
	out := make([]uint16, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialUint16s) apply(result simd.Uint16s) PartialUint16s {
	return PartialUint16s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialUint16s) Add(other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
//
// Subtraction wraps modulo 65536, exactly as Go's uint16 arithmetic does — going
// below zero underflows to a large value. The stdlib's saturating alternative is the
// separate SubSaturated method, which this deliberately does not use.
func (p PartialUint16s) Sub(other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies by other lane-wise, leaving padded lanes untouched.
func (p PartialUint16s) Mul(other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Mul(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
func (p PartialUint16s) Min(other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Min(other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
func (p PartialUint16s) Max(other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Max(other.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func (p PartialUint16s) Clamp(lower, upper PartialUint16s) PartialUint16s {
	return p.apply(p.vec.Max(lower.vec).Min(upper.vec))
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialUint16s) Sum() uint16 {
	var buf [maxLanes]uint16
	n := p.StorePart(buf[:])

	total := uint16(0)
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
// value present" only because BroadcastUint16 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsUint16
// to collapse a whole stream to a single bool.
func (p PartialUint16s) Contains(target PartialUint16s) simd.Mask16s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialUint16s) Select(mask simd.Mask16s, other PartialUint16s) PartialUint16s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsUint16 uses it; the
// exported Contains stays element-wise.
//
// The scratch buffer is int16, not uint16: mask conversion methods are named for the
// signed type of the width, so Mask16s.ToInt16s yields simd.Int16s whatever element
// type produced the mask.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialUint16s) anyMatch(target PartialUint16s) bool {
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
func (p PartialUint16s) rawLane(index int) uint16 {
	var buf [maxLanes]uint16
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
}
