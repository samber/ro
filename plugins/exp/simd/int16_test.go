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

func lanesInt16() int { return simd.BroadcastInt16s(0).Len() }

// sizeSweepInt16 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt16() []int {
	lanes := lanesInt16()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt16 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampInt16(size int) []int16 {
	out := make([]int16, size)
	for i := range out {
		out[i] = int16(i%1000 + 1)
	}

	return out
}

// collectInt16Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt16s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt16Values[V Int16Buffer[V]](t *testing.T, input []int16, operators ...func(ro.Observable[V]) ro.Observable[V]) []int16 {
	t.Helper()

	vectors := VectorizeInt16[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int16, int16](
			vectors,
			ro.Map(func(v V) []int16 {
				var buffer [maxLanes]int16
				n := v.StorePart(buffer[:])

				return append([]int16{}, buffer[:n]...)
			}),
			ro.Flatten[int16](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt16 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt16(mask simd.Mask16s) []bool {
	lanes := lanesInt16()

	var buf [maxLanes]int16
	mask.ToInt16s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripInt16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		got := collectInt16Values[PartialInt16s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesInt16(t *testing.T) {
	t.Parallel()

	lanes := lanesInt16()

	// 35 items at 8 lanes is four full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeInt16[PartialInt16s](ro.FromSlice(rampInt16(35))))
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

func TestAddInt16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int16) int16 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectInt16Values[PartialInt16s](t, input, AddInt16(BroadcastInt16(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubInt16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int16) int16 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectInt16Values[PartialInt16s](t, input, SubInt16(BroadcastInt16(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampInt16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[int16](10, 50)),
		)
		assert.NoError(t, err)

		got := collectInt16Values[PartialInt16s](t, input, ClampInt16(BroadcastInt16(10), BroadcastInt16(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampInt16InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []int16{1, 30, 80}

	got := collectInt16Values[PartialInt16s](t, input, ClampInt16(BroadcastInt16(50), BroadcastInt16(10)))

	assert.Equal(t, []int16{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[int16](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsInt16ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialInt16s{}.LoadPart([]int16{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastInt16(7)), BroadcastInt16(0))

	assert.Equal(t, []int16{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsInt16MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialInt16s{}.LoadPart([]int16{0, 1, 0}, 3)

	lanes := maskLanesInt16(vector.Contains(BroadcastInt16(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedInt16(t *testing.T) {
	t.Parallel()

	input := rampInt16(3)

	got := collectInt16Values[PartialInt16s](t, input, AddInt16(BroadcastInt16(100)))

	assert.Equal(t, []int16{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsInt16(t *testing.T) {
	t.Parallel()

	lanes := lanesInt16()

	tail := PartialInt16s{}.LoadPart([]int16{1, 2, 3}, 3).Add(BroadcastInt16(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, int16(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumInt16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[int16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, int16](ro.FromSlice(input), VectorizeInt16, ReduceSumInt16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinInt16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[int16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, int16](ro.FromSlice(input), VectorizeInt16, ReduceMinInt16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxInt16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampInt16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[int16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, int16](ro.FromSlice(input), VectorizeInt16, ReduceMaxInt16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsInt16ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[int16, PartialInt16s, bool](
			ro.FromSlice([]int16{1, 2, 3}),
			VectorizeInt16,
			ReduceContainsInt16(BroadcastInt16(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsInt16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		for _, target := range []int16{1, 3, 100, 32767} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[int16, PartialInt16s, bool](
					ro.FromSlice(input),
					VectorizeInt16,
					ReduceContainsInt16(BroadcastInt16(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithInt16ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt16[PartialInt16s](ro.FromSlice(rampInt16(20)))
	right := VectorizeInt16[PartialInt16s](ro.FromSlice(rampInt16(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt16s, []int16, int16](
			AddWithInt16(right)(left),
			ro.Map(func(v PartialInt16s) []int16 { return v.Values() }),
			ro.Flatten[int16](),
		),
	)
	assert.NoError(t, err)

	want := make([]int16, 20)
	for i, v := range rampInt16(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int16, PartialInt16s, []int16, int16](
			ro.FromSlice(rampInt16(10)),
			VectorizeInt16,
			ro.Map(func(v PartialInt16s) []int16 {
				return v.Add(BroadcastInt16(42)).Min(BroadcastInt16(50)).Values()
			}),
			ro.Flatten[int16](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int16{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsInt16(t *testing.T) {
	t.Parallel()

	lanes := lanesInt16()
	batch := rampInt16(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt16s(batch)), AddInt16(simd.BroadcastInt16s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]int16
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesInt16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[int16, PartialInt16s, int16](ro.Empty[int16](), VectorizeInt16, ReduceSumInt16),
	)
	assert.NoError(t, err)
	assert.Equal(t, []int16{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeInt16[PartialInt16s](ro.Empty[int16]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialInt16s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeInt16[PartialInt16s](ro.Throw[int16](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[int16, PartialInt16s, int16](ro.Throw[int16](assert.AnError), VectorizeInt16, ReduceSumInt16),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
