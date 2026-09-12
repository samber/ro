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

// Uint8Vector describes the operations available on any uint8 vector, whether it is
// the standard library's simd.Uint8s or this package's PartialUint8s.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastUint8s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastUint8.
type Uint8Vector[V any] interface {
	LaneStore[uint8]

	Add(V) V
	Sub(V) V
	Mul(V) V
	Min(V) V
	Max(V) V
}

// Uint8Buffer is a Uint8Vector that can also be built from a slice of scalars. Only
// PartialUint8s satisfies it.
type Uint8Buffer[V any] interface {
	Uint8Vector[V]
	LaneBuffer[V, uint8]
}

// Uint8Searchable is a Uint8Vector that can test its own lanes for a value and
// widen a scalar to search for. Only
// PartialUint8s satisfies it.
type Uint8Searchable[V any] interface {
	Uint8Vector[V]
	LaneMatcher[V]
	Broadcast(uint8) V
}

// PartialUint8s is a vector of uint8 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialUint8s struct {
	vec  simd.Uint8s
	mask simd.Mask8s
	n    int
}

// BroadcastUint8 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastUint8(value uint8) PartialUint8s {
	vec := simd.BroadcastUint8s(value)

	return PartialUint8s{vec: vec, mask: fullMaskUint8(), n: vec.Len()}
}

// Broadcast is the method form of BroadcastUint8, so the Searchable constraint can
// widen a scalar from inside a generic operator body — method dispatch through the
// constraint is the one call shape the simd specializer handles there. The
// receiver carries no state.
func (p PartialUint8s) Broadcast(value uint8) PartialUint8s {
	return BroadcastUint8(value)
}

// fullMaskUint8 is an all-true mask. simd.Uint8s has no ToMask, but masks are per
// bit-width and therefore interchangeable between the signed and unsigned type of
// that width, so build it through simd.Int8s. ToMask compares against zero, so
// broadcasting any non-zero value yields every lane set.
func fullMaskUint8() simd.Mask8s { return simd.BroadcastInt8s(1).ToMask() }

// prefixMaskUint8 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, and simd.Uint8s has neither Less nor ToMask, so
// compare a ramp against n through simd.Int8s — the resulting Mask8s is shared by
// both types of the 8-bit width.
func prefixMaskUint8(n int) simd.Mask8s {
	lanes := simd.BroadcastUint8s(0).Len()

	var ramp [maxLanes]int8
	for i := range ramp {
		ramp[i] = int8(i)
	}

	return simd.LoadInt8s(ramp[:lanes]).Less(simd.BroadcastInt8s(int8(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialUint8s) Load(src []uint8) PartialUint8s {
	vec := simd.LoadUint8s(src)

	return PartialUint8s{vec: vec, mask: fullMaskUint8(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialUint8s) LoadPart(src []uint8, n int) PartialUint8s {
	lanes := simd.BroadcastUint8s(0).Len()

	var buf [maxLanes]uint8
	copy(buf[:], src[:n])

	return PartialUint8s{vec: simd.LoadUint8s(buf[:lanes]), mask: prefixMaskUint8(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialUint8s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialUint8s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialUint8s) StorePart(dst []uint8) int {
	var buf [maxLanes]uint8
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialUint8s) Values() []uint8 {
	out := make([]uint8, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialUint8s) apply(result simd.Uint8s) PartialUint8s {
	return PartialUint8s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialUint8s) Add(other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
//
// Subtraction wraps modulo 256, exactly as Go's uint8 arithmetic does — going below
// zero underflows to a large value. The stdlib's saturating alternative is the
// separate SubSaturated method, which this deliberately does not use.
func (p PartialUint8s) Sub(other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Sub(other.vec))
}

// Mul multiplies by other lane-wise, leaving padded lanes untouched.
func (p PartialUint8s) Mul(other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Mul(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
func (p PartialUint8s) Min(other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Min(other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
func (p PartialUint8s) Max(other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Max(other.vec))
}

// Clamp bounds every valid lane to [lower, upper].
//
// Passing lower > upper is a programmer error. Unlike core ro.Clamp, which rejects
// it at construction with a panic, this cannot detect it: the bounds are opaque
// vectors that a generic operator may not inspect. The composition is Max(lower)
// then Min(upper), so inverted bounds collapse every lane to upper.
func (p PartialUint8s) Clamp(lower, upper PartialUint8s) PartialUint8s {
	return p.apply(p.vec.Max(lower.vec).Min(upper.vec))
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialUint8s) Sum() uint8 {
	var buf [maxLanes]uint8
	n := p.StorePart(buf[:])

	total := uint8(0)
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
// value present" only because BroadcastUint8 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsUint8
// to collapse a whole stream to a single bool.
func (p PartialUint8s) Contains(target PartialUint8s) simd.Mask8s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialUint8s) Select(mask simd.Mask8s, other PartialUint8s) PartialUint8s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsUint8 uses it; the
// exported Contains stays element-wise.
//
// The scratch buffer is int8, not uint8: mask conversion methods are named for the
// signed type of the width, so Mask8s.ToInt8s yields simd.Int8s whatever element
// type produced the mask.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
func (p PartialUint8s) anyMatch(target PartialUint8s) bool {
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
func (p PartialUint8s) rawLane(index int) uint8 {
	var buf [maxLanes]uint8
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
}
