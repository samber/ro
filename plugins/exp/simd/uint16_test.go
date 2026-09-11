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
	"slices"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func lanesUint16() int { return simd.BroadcastUint16s(0).Len() }

// sizeSweepUint16 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint16() []int {
	lanes := lanesUint16()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint16 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampUint16(size int) []uint16 {
	out := make([]uint16, size)
	for i := range out {
		out[i] = uint16(i%1000 + 1)
	}

	return out
}

// collectUint16Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint16s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint16Values[V Uint16Buffer[V]](t *testing.T, input []uint16, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint16 {
	t.Helper()

	vectors := VectorizeUint16[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint16, uint16](
			vectors,
			ro.Map(func(v V) []uint16 {
				var buffer [maxLanes]uint16
				n := v.StorePart(buffer[:])

				return append([]uint16{}, buffer[:n]...)
			}),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint16 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint16(mask simd.Mask16s) []bool {
	lanes := lanesUint16()

	var buf [maxLanes]int16
	mask.ToInt16s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		got := collectUint16Values[PartialUint16s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()

	// 35 items at 8 lanes is four full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint16[PartialUint16s](ro.FromSlice(rampUint16(35))))
	assert.NoError(t, err)

	counts := make([]int, 0, len(vectors))
	for _, v := range vectors {
		counts = append(counts, v.Count())
	}

	expected := []int{}
	for remaining := 35; remaining > 0; remaining -= lanes {
		expected = append(expected, min(lanes, remaining))
	}

	assert.Equal(t, expected, counts)
}

func TestAddUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint16) uint16 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectUint16Values[PartialUint16s](t, input, AddUint16(BroadcastUint16(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint16) uint16 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectUint16Values[PartialUint16s](t, input, SubUint16(BroadcastUint16(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[uint16](10, 50)),
		)
		assert.NoError(t, err)

		got := collectUint16Values[PartialUint16s](t, input, ClampUint16(BroadcastUint16(10), BroadcastUint16(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampUint16InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []uint16{1, 30, 80}

	got := collectUint16Values[PartialUint16s](t, input, ClampUint16(BroadcastUint16(50), BroadcastUint16(10)))

	assert.Equal(t, []uint16{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[uint16](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint16ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint16s{}.LoadPart([]uint16{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint16(7)), BroadcastUint16(0))

	assert.Equal(t, []uint16{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsUint16MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint16s{}.LoadPart([]uint16{0, 1, 0}, 3)

	lanes := maskLanesUint16(vector.Contains(BroadcastUint16(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint16(t *testing.T) {
	t.Parallel()

	input := rampUint16(3)

	got := collectUint16Values[PartialUint16s](t, input, AddUint16(BroadcastUint16(100)))

	assert.Equal(t, []uint16{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()

	tail := PartialUint16s{}.LoadPart([]uint16{1, 2, 3}, 3).Add(BroadcastUint16(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint16(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceSumUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMinUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMaxUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint16ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, bool](
			ro.FromSlice([]uint16{1, 2, 3}),
			VectorizeUint16,
			ReduceContainsUint16(BroadcastUint16(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		for _, target := range []uint16{1, 3, 100, 65535} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint16, PartialUint16s, bool](
					ro.FromSlice(input),
					VectorizeUint16,
					ReduceContainsUint16(BroadcastUint16(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithUint16ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint16[PartialUint16s](ro.FromSlice(rampUint16(20)))
	right := VectorizeUint16[PartialUint16s](ro.FromSlice(rampUint16(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint16s, []uint16, uint16](
			AddWithUint16(right)(left),
			ro.Map(func(v PartialUint16s) []uint16 { return v.Values() }),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	want := make([]uint16, 20)
	for i, v := range rampUint16(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint16, PartialUint16s, []uint16, uint16](
			ro.FromSlice(rampUint16(10)),
			VectorizeUint16,
			ro.Map(func(v PartialUint16s) []uint16 {
				return v.Add(BroadcastUint16(42)).Min(BroadcastUint16(50)).Values()
			}),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint16{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()
	batch := rampUint16(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint16s(batch)), AddUint16(simd.BroadcastUint16s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint16
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesUint16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.Empty[uint16](), VectorizeUint16, ReduceSumUint16),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint16{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint16[PartialUint16s](ro.Empty[uint16]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint16s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint16[PartialUint16s](ro.Throw[uint16](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.Throw[uint16](assert.AnError), VectorizeUint16, ReduceSumUint16),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

// Masks are built through the signed vector type of the same width, which is the one
// place a signed comparison could leak into unsigned arithmetic. 40000 has its high
// bit set: read as int16 it is negative, so every ordering below would come out
// backwards.
func TestMinMaxUint16TreatsHighBitAsLarge(t *testing.T) {
	t.Parallel()

	input := []uint16{1000, 40000, 20000}

	capped := collectUint16Values[PartialUint16s](t, input, MinUint16(BroadcastUint16(30000)))
	assert.Equal(t, []uint16{1000, 30000, 20000}, capped,
		"40000 must be clamped down, not treated as smaller than 1000")

	raised := collectUint16Values[PartialUint16s](t, input, MaxUint16(BroadcastUint16(5000)))
	assert.Equal(t, []uint16{5000, 40000, 20000}, raised)

	largest, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMaxUint16),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint16{40000}, largest)

	smallest, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMinUint16),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint16{1000}, smallest)
}
