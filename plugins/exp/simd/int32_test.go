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

func lanesInt32() int { return simd.BroadcastInt32s(0).Len() }

// sizeSweepInt32 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt32() []int {
	lanes := lanesInt32()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt32 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampInt32(size int) []int32 {
	out := make([]int32, size)
	for i := range out {
		out[i] = int32(i%1000 + 1)
	}

	return out
}

// collectInt32Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt32s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt32Values[V Int32Buffer[V]](t *testing.T, input []int32, operators ...func(ro.Observable[V]) ro.Observable[V]) []int32 {
	t.Helper()

	vectors := VectorizeInt32[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int32, int32](
			vectors,
			ro.Map(func(v V) []int32 {
				var buffer [maxLanes]int32
				n := v.StorePart(buffer[:])

				return append([]int32{}, buffer[:n]...)
			}),
			ro.Flatten[int32](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt32 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt32(mask simd.Mask32s) []bool {
	lanes := lanesInt32()

	var buf [maxLanes]int32
	mask.ToInt32s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripInt32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		got := collectInt32Values[PartialInt32s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesInt32(t *testing.T) {
	t.Parallel()

	lanes := lanesInt32()

	// 35 items at 4 lanes is eight full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeInt32[PartialInt32s](ro.FromSlice(rampInt32(35))))
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

func TestAddInt32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int32) int32 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectInt32Values[PartialInt32s](t, input, AddInt32(BroadcastInt32(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubInt32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int32) int32 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectInt32Values[PartialInt32s](t, input, SubInt32(BroadcastInt32(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampInt32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[int32](10, 50)),
		)
		assert.NoError(t, err)

		got := collectInt32Values[PartialInt32s](t, input, ClampInt32(BroadcastInt32(10), BroadcastInt32(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampInt32InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []int32{1, 30, 80}

	got := collectInt32Values[PartialInt32s](t, input, ClampInt32(BroadcastInt32(50), BroadcastInt32(10)))

	assert.Equal(t, []int32{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[int32](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsInt32ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialInt32s{}.LoadPart([]int32{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastInt32(7)), BroadcastInt32(0))

	assert.Equal(t, []int32{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsInt32MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialInt32s{}.LoadPart([]int32{0, 1, 0}, 3)

	lanes := maskLanesInt32(vector.Contains(BroadcastInt32(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedInt32(t *testing.T) {
	t.Parallel()

	input := rampInt32(3)

	got := collectInt32Values[PartialInt32s](t, input, AddInt32(BroadcastInt32(100)))

	assert.Equal(t, []int32{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsInt32(t *testing.T) {
	t.Parallel()

	lanes := lanesInt32()

	tail := PartialInt32s{}.LoadPart([]int32{1, 2, 3}, 3).Add(BroadcastInt32(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, int32(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumInt32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[int32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, int32](ro.FromSlice(input), VectorizeInt32, ReduceSumInt32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinInt32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[int32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, int32](ro.FromSlice(input), VectorizeInt32, ReduceMinInt32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxInt32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampInt32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[int32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, int32](ro.FromSlice(input), VectorizeInt32, ReduceMaxInt32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsInt32ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[int32, PartialInt32s, bool](
			ro.FromSlice([]int32{1, 2, 3}),
			VectorizeInt32,
			ReduceContainsInt32(BroadcastInt32(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsInt32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		for _, target := range []int32{1, 3, 100, 2147483647} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[int32, PartialInt32s, bool](
					ro.FromSlice(input),
					VectorizeInt32,
					ReduceContainsInt32(BroadcastInt32(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithInt32ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt32[PartialInt32s](ro.FromSlice(rampInt32(20)))
	right := VectorizeInt32[PartialInt32s](ro.FromSlice(rampInt32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt32s, []int32, int32](
			AddWithInt32(right)(left),
			ro.Map(func(v PartialInt32s) []int32 { return v.Values() }),
			ro.Flatten[int32](),
		),
	)
	assert.NoError(t, err)

	want := make([]int32, 20)
	for i, v := range rampInt32(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int32, PartialInt32s, []int32, int32](
			ro.FromSlice(rampInt32(10)),
			VectorizeInt32,
			ro.Map(func(v PartialInt32s) []int32 {
				return v.Add(BroadcastInt32(42)).Min(BroadcastInt32(50)).Values()
			}),
			ro.Flatten[int32](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int32{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsInt32(t *testing.T) {
	t.Parallel()

	lanes := lanesInt32()
	batch := rampInt32(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt32s(batch)), AddInt32(simd.BroadcastInt32s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]int32
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesInt32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[int32, PartialInt32s, int32](ro.Empty[int32](), VectorizeInt32, ReduceSumInt32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []int32{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeInt32[PartialInt32s](ro.Empty[int32]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialInt32s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeInt32[PartialInt32s](ro.Throw[int32](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[int32, PartialInt32s, int32](ro.Throw[int32](assert.AnError), VectorizeInt32, ReduceSumInt32),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
