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

func lanesUint32() int { return simd.BroadcastUint32s(0).Len() }

// sizeSweepUint32 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint32() []int {
	lanes := lanesUint32()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint32 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampUint32(size int) []uint32 {
	out := make([]uint32, size)
	for i := range out {
		out[i] = uint32(i%1000 + 1)
	}

	return out
}

// collectUint32Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint32s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint32Values[V Uint32Buffer[V]](t *testing.T, input []uint32, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint32 {
	t.Helper()

	vectors := VectorizeUint32[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint32, uint32](
			vectors,
			ro.Map(func(v V) []uint32 {
				var buffer [maxLanes]uint32
				n := v.StorePart(buffer[:])

				return append([]uint32{}, buffer[:n]...)
			}),
			ro.Flatten[uint32](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint32 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint32(mask simd.Mask32s) []bool {
	lanes := lanesUint32()

	var buf [maxLanes]int32
	mask.ToInt32s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		got := collectUint32Values[PartialUint32s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()

	// 35 items at 4 lanes is eight full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint32[PartialUint32s](ro.FromSlice(rampUint32(35))))
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

func TestAddUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint32) uint32 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectUint32Values[PartialUint32s](t, input, AddUint32(BroadcastUint32(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint32) uint32 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectUint32Values[PartialUint32s](t, input, SubUint32(BroadcastUint32(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[uint32](10, 50)),
		)
		assert.NoError(t, err)

		got := collectUint32Values[PartialUint32s](t, input, ClampUint32(BroadcastUint32(10), BroadcastUint32(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampUint32InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []uint32{1, 30, 80}

	got := collectUint32Values[PartialUint32s](t, input, ClampUint32(BroadcastUint32(50), BroadcastUint32(10)))

	assert.Equal(t, []uint32{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[uint32](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint32ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint32s{}.LoadPart([]uint32{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint32(7)), BroadcastUint32(0))

	assert.Equal(t, []uint32{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsUint32MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint32s{}.LoadPart([]uint32{0, 1, 0}, 3)

	lanes := maskLanesUint32(vector.Contains(BroadcastUint32(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint32(t *testing.T) {
	t.Parallel()

	input := rampUint32(3)

	got := collectUint32Values[PartialUint32s](t, input, AddUint32(BroadcastUint32(100)))

	assert.Equal(t, []uint32{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()

	tail := PartialUint32s{}.LoadPart([]uint32{1, 2, 3}, 3).Add(BroadcastUint32(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint32(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceSumUint32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMinUint32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMaxUint32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint32ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, bool](
			ro.FromSlice([]uint32{1, 2, 3}),
			VectorizeUint32,
			ReduceContainsUint32(BroadcastUint32(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		for _, target := range []uint32{1, 3, 100, 4294967295} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint32, PartialUint32s, bool](
					ro.FromSlice(input),
					VectorizeUint32,
					ReduceContainsUint32(BroadcastUint32(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithUint32ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint32[PartialUint32s](ro.FromSlice(rampUint32(20)))
	right := VectorizeUint32[PartialUint32s](ro.FromSlice(rampUint32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint32s, []uint32, uint32](
			AddWithUint32(right)(left),
			ro.Map(func(v PartialUint32s) []uint32 { return v.Values() }),
			ro.Flatten[uint32](),
		),
	)
	assert.NoError(t, err)

	want := make([]uint32, 20)
	for i, v := range rampUint32(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint32, PartialUint32s, []uint32, uint32](
			ro.FromSlice(rampUint32(10)),
			VectorizeUint32,
			ro.Map(func(v PartialUint32s) []uint32 {
				return v.Add(BroadcastUint32(42)).Min(BroadcastUint32(50)).Values()
			}),
			ro.Flatten[uint32](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint32{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()
	batch := rampUint32(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint32s(batch)), AddUint32(simd.BroadcastUint32s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint32
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesUint32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.Empty[uint32](), VectorizeUint32, ReduceSumUint32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint32{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint32[PartialUint32s](ro.Empty[uint32]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint32s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint32[PartialUint32s](ro.Throw[uint32](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.Throw[uint32](assert.AnError), VectorizeUint32, ReduceSumUint32),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

// Masks are built through the signed vector type of the same width, which is the one
// place a signed comparison could leak into unsigned arithmetic. 3000000000 has its high
// bit set: read as int32 it is negative, so every ordering below would come out
// backwards.
func TestMinMaxUint32TreatsHighBitAsLarge(t *testing.T) {
	t.Parallel()

	input := []uint32{100000, 3000000000, 2000000000}

	capped := collectUint32Values[PartialUint32s](t, input, MinUint32(BroadcastUint32(2500000000)))
	assert.Equal(t, []uint32{100000, 2500000000, 2000000000}, capped,
		"3000000000 must be clamped down, not treated as smaller than 100000")

	raised := collectUint32Values[PartialUint32s](t, input, MaxUint32(BroadcastUint32(500000)))
	assert.Equal(t, []uint32{500000, 3000000000, 2000000000}, raised)

	largest, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMaxUint32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint32{3000000000}, largest)

	smallest, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMinUint32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint32{100000}, smallest)
}
