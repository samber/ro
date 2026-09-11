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

func lanesUint8() int { return simd.BroadcastUint8s(0).Len() }

// sizeSweepUint8 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint8() []int {
	lanes := lanesUint8()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint8 builds 1, 2, 3... wrapping within uint8 so long inputs stay in range.
func rampUint8(size int) []uint8 {
	out := make([]uint8, size)
	for i := range out {
		out[i] = uint8(i%100 + 1)
	}

	return out
}

// collectUint8Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint8s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint8Values[V Uint8Buffer[V]](t *testing.T, input []uint8, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint8 {
	t.Helper()

	vectors := VectorizeUint8[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint8, uint8](
			vectors,
			ro.Map(func(v V) []uint8 {
				var buffer [maxLanes]uint8
				n := v.StorePart(buffer[:])

				return append([]uint8{}, buffer[:n]...)
			}),
			ro.Flatten[uint8](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint8 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint8(mask simd.Mask8s) []bool {
	lanes := lanesUint8()

	var buf [maxLanes]int8
	mask.ToInt8s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripUint8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		got := collectUint8Values[PartialUint8s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()

	// 35 items at 16 lanes is two full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint8[PartialUint8s](ro.FromSlice(rampUint8(35))))
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

func TestAddUint8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint8) uint8 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectUint8Values[PartialUint8s](t, input, AddUint8(BroadcastUint8(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubUint8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint8) uint8 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectUint8Values[PartialUint8s](t, input, SubUint8(BroadcastUint8(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampUint8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[uint8](10, 50)),
		)
		assert.NoError(t, err)

		got := collectUint8Values[PartialUint8s](t, input, ClampUint8(BroadcastUint8(10), BroadcastUint8(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampUint8InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []uint8{1, 30, 80}

	got := collectUint8Values[PartialUint8s](t, input, ClampUint8(BroadcastUint8(50), BroadcastUint8(10)))

	assert.Equal(t, []uint8{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[uint8](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint8ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint8s{}.LoadPart([]uint8{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint8(7)), BroadcastUint8(0))

	assert.Equal(t, []uint8{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsUint8MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint8s{}.LoadPart([]uint8{0, 1, 0}, 3)

	lanes := maskLanesUint8(vector.Contains(BroadcastUint8(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint8(t *testing.T) {
	t.Parallel()

	input := rampUint8(3)

	got := collectUint8Values[PartialUint8s](t, input, AddUint8(BroadcastUint8(100)))

	assert.Equal(t, []uint8{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()

	tail := PartialUint8s{}.LoadPart([]uint8{1, 2, 3}, 3).Add(BroadcastUint8(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint8(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumUint8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, ReduceSumUint8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinUint8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, ReduceMinUint8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxUint8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, ReduceMaxUint8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint8ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, bool](
			ro.FromSlice([]uint8{1, 2, 3}),
			VectorizeUint8,
			ReduceContainsUint8(BroadcastUint8(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		for _, target := range []uint8{1, 3, 100, 255} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint8, PartialUint8s, bool](
					ro.FromSlice(input),
					VectorizeUint8,
					ReduceContainsUint8(BroadcastUint8(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithUint8ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint8[PartialUint8s](ro.FromSlice(rampUint8(20)))
	right := VectorizeUint8[PartialUint8s](ro.FromSlice(rampUint8(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint8s, []uint8, uint8](
			AddWithUint8(right)(left),
			ro.Map(func(v PartialUint8s) []uint8 { return v.Values() }),
			ro.Flatten[uint8](),
		),
	)
	assert.NoError(t, err)

	want := make([]uint8, 20)
	for i, v := range rampUint8(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint8(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint8, PartialUint8s, []uint8, uint8](
			ro.FromSlice(rampUint8(10)),
			VectorizeUint8,
			ro.Map(func(v PartialUint8s) []uint8 {
				return v.Add(BroadcastUint8(42)).Min(BroadcastUint8(50)).Values()
			}),
			ro.Flatten[uint8](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint8{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()
	batch := rampUint8(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint8s(batch)), AddUint8(simd.BroadcastUint8s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint8
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesUint8(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.Empty[uint8](), VectorizeUint8, ReduceSumUint8),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint8{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint8[PartialUint8s](ro.Empty[uint8]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint8s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint8[PartialUint8s](ro.Throw[uint8](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.Throw[uint8](assert.AnError), VectorizeUint8, ReduceSumUint8),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

// Masks are built through the signed vector type of the same width, which is the one
// place a signed comparison could leak into unsigned arithmetic. 200 has its high
// bit set: read as int8 it is negative, so every ordering below would come out
// backwards.
func TestMinMaxUint8TreatsHighBitAsLarge(t *testing.T) {
	t.Parallel()

	input := []uint8{100, 200, 150}

	capped := collectUint8Values[PartialUint8s](t, input, MinUint8(BroadcastUint8(180)))
	assert.Equal(t, []uint8{100, 180, 150}, capped,
		"200 must be clamped down, not treated as smaller than 100")

	raised := collectUint8Values[PartialUint8s](t, input, MaxUint8(BroadcastUint8(120)))
	assert.Equal(t, []uint8{120, 200, 150}, raised)

	largest, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, ReduceMaxUint8),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint8{200}, largest)

	smallest, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, ReduceMinUint8),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint8{100}, smallest)
}
