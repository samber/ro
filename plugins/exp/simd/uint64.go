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

// Uint64Vector describes the operations available on an uint64 vector.
//
// Every method takes and returns V rather than a scalar. SIMD has no scalar-operand
// arithmetic, and a generic function body may not call simd.BroadcastUint64s to widen
// one — doing so makes the function simd-dependent and breaks the compiler's own
// clone dispatcher. Callers widen scalars themselves with BroadcastUint64.
//
// Unlike Int8Vector there is no Mul: the standard library provides no 64-bit lane
// multiply, so no uint64 vector type can offer one.
//
// simd.Uint64s does not satisfy this interface either — it has no Min or Max methods —
// so in practice only PartialUint64s, which synthesizes both from Less and IfElse,
// flows through the operators.
type Uint64Vector[V any] interface {
	LaneStore[uint64]

	Add(V) V
	Sub(V) V
	Min(V) V
	Max(V) V
}

// Uint64Buffer is an Uint64Vector that can also be built from a slice of scalars. Only
// PartialUint64s satisfies it.
type Uint64Buffer[V any] interface {
	Uint64Vector[V]
	LaneBuffer[V, uint64]
}

// Uint64Searchable is an Uint64Vector that can test its own lanes for a value. Only
// PartialUint64s satisfies it.
type Uint64Searchable[V any] interface {
	Uint64Vector[V]
	LaneMatcher[V]
}

// PartialUint64s is a vector of uint64 lanes where only the first n are valid.
//
// A stream rarely delivers a multiple of the lane width, so the final vector of a
// batch is short. Rather than dropping those values or padding them into the
// result, every operation applies mask so padded lanes keep whatever they held
// before — zero, as LoadPart leaves them.
//
// mask is always the contiguous prefix described by n; the two are set together at
// construction and preserved by every operation, so they cannot drift apart.
type PartialUint64s struct {
	vec  simd.Uint64s
	mask simd.Mask64s
	n    int
}

// BroadcastUint64 returns a full vector with value in every lane. Use it to widen a
// scalar into an operand for the element-wise operators.
func BroadcastUint64(value uint64) PartialUint64s {
	vec := simd.BroadcastUint64s(value)

	return PartialUint64s{vec: vec, mask: fullMaskUint64(), n: vec.Len()}
}

// fullMaskUint64 is an all-true mask. simd.Uint64s has no ToMask, so it is built
// through simd.Int64s — the resulting Mask64s is shared by both types of the 64-bit
// width. ToMask compares against zero, so broadcasting any non-zero value yields
// every lane set.
func fullMaskUint64() simd.Mask64s { return simd.BroadcastInt64s(1).ToMask() }

// prefixMaskUint64 builds a mask whose first n lanes are true. simd offers no
// count-based mask constructor, so compare a ramp vector against n.
//
// Unlike the narrower unsigned types, simd.Uint64s has Less, so the comparison stays
// in the unsigned domain rather than borrowing the signed type's.
func prefixMaskUint64(n int) simd.Mask64s {
	lanes := simd.BroadcastUint64s(0).Len()

	var ramp [maxLanes]uint64
	for i := range ramp {
		ramp[i] = uint64(i)
	}

	return simd.LoadUint64s(ramp[:lanes]).Less(simd.BroadcastUint64s(uint64(n)))
}

// Load builds a full vector from exactly Len() scalars.
func (p PartialUint64s) Load(src []uint64) PartialUint64s {
	vec := simd.LoadUint64s(src)

	return PartialUint64s{vec: vec, mask: fullMaskUint64(), n: vec.Len()}
}

// LoadPart builds a vector whose first n lanes are valid, zero-filling the rest.
func (p PartialUint64s) LoadPart(src []uint64, n int) PartialUint64s {
	lanes := simd.BroadcastUint64s(0).Len()

	var buf [maxLanes]uint64
	copy(buf[:], src[:n])

	return PartialUint64s{vec: simd.LoadUint64s(buf[:lanes]), mask: prefixMaskUint64(n), n: n}
}

// Len reports the lane capacity, not the number of valid lanes. It is meaningful on
// the zero value, which is how the operators discover the architecture's width.
func (p PartialUint64s) Len() int { return p.vec.Len() }

// Count reports how many lanes hold valid data.
func (p PartialUint64s) Count() int { return p.n }

// StorePart writes the valid lanes into dst and returns how many it wrote.
func (p PartialUint64s) StorePart(dst []uint64) int {
	var buf [maxLanes]uint64
	p.vec.Store(buf[:p.vec.Len()])

	n := min(p.n, len(dst))
	copy(dst[:n], buf[:n])

	return n
}

// Values returns the valid lanes as a freshly allocated slice. Prefer StorePart in
// hot paths; this exists so pipelines can hand a slice straight to ro.Flatten.
func (p PartialUint64s) Values() []uint64 {
	out := make([]uint64, p.n)
	p.StorePart(out)

	return out
}

// apply rebuilds the vector from a computed result, restoring the original value in
// every padded lane. Every element-wise method funnels through here, so masking can
// never be forgotten on one of them.
func (p PartialUint64s) apply(result simd.Uint64s) PartialUint64s {
	return PartialUint64s{vec: result.IfElse(p.mask, p.vec), mask: p.mask, n: p.n}
}

// Add adds other lane-wise, leaving padded lanes untouched.
func (p PartialUint64s) Add(other PartialUint64s) PartialUint64s {
	return p.apply(p.vec.Add(other.vec))
}

// Sub subtracts other lane-wise, leaving padded lanes untouched.
func (p PartialUint64s) Sub(other PartialUint64s) PartialUint64s {
	return p.apply(p.vec.Sub(other.vec))
}

// Min keeps the smaller of each lane pair, leaving padded lanes untouched.
//
// simd.Uint64s has no Min method, so it is synthesized from Less and IfElse:
// x.IfElse(mask, y) keeps x where mask is set, so selecting on x.Less(y) keeps x
// exactly in the lanes where it is the smaller.
func (p PartialUint64s) Min(other PartialUint64s) PartialUint64s {
	return p.apply(p.vec.IfElse(p.vec.Less(other.vec), other.vec))
}

// Max keeps the larger of each lane pair, leaving padded lanes untouched.
//
// Synthesized like Min, with the IfElse operands swapped: where p is less than
// other, keep other.
func (p PartialUint64s) Max(other PartialUint64s) PartialUint64s {
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
func (p PartialUint64s) Clamp(lower, upper PartialUint64s) PartialUint64s {
	return p.Max(lower).Min(upper)
}

// Sum totals the valid lanes. simd has no horizontal reduction, so this stores the
// vector and folds it in scalar code — the only way to collapse lanes to a value.
func (p PartialUint64s) Sum() uint64 {
	var buf [maxLanes]uint64
	n := p.StorePart(buf[:])

	total := uint64(0)
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
// value present" only because BroadcastUint64 puts that one value in every lane of
// target; a non-uniform target compares position by position instead.
//
// Feed the mask back into Select to act on the matches, or use ReduceContainsUint64
// to collapse a whole stream to a single bool.
func (p PartialUint64s) Contains(target PartialUint64s) simd.Mask64s {
	return p.vec.Equal(target.vec).And(p.mask)
}

// Select takes each lane from p where mask is set and from other where it is not,
// leaving padded lanes untouched. It is the counterpart to Contains, whose mask can
// be handed straight back in.
func (p PartialUint64s) Select(mask simd.Mask64s, other PartialUint64s) PartialUint64s {
	return p.apply(p.vec.IfElse(mask, other.vec))
}

// anyMatch folds Contains' mask down to a single answer in scalar code, since simd
// offers no way to extract a mask as a bitmask. ReduceContainsUint64 uses it; the
// exported Contains stays element-wise.
//
// Scanning every lane, not just the first n, is deliberate: it leaves the mask solely
// responsible for excluding padding, so a mistake there shows up as a wrong answer
// rather than being hidden by a second bound.
// The buffer is signed because a mask converts only to the signed vector type of its
// width; only the zero/non-zero distinction is read from it.
func (p PartialUint64s) anyMatch(target PartialUint64s) bool {
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
func (p PartialUint64s) rawLane(index int) uint64 {
	var buf [maxLanes]uint64
	p.vec.Store(buf[:p.vec.Len()])

	return buf[index]
}
