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

func lanesInt8() int { return simd.BroadcastInt8s(0).Len() }

// sizeSweepInt8 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt8() []int {
	lanes := lanesInt8()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt8 builds 1, 2, 3... wrapping within int8 so long inputs stay in range.
func rampInt8(size int) []int8 {
	out := make([]int8, size)
	for i := range out {
		out[i] = int8(i%100 + 1)
	}

	return out
}

// collectInt8Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt8s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt8Values[V Int8Buffer[V]](t *testing.T, input []int8, operators ...func(ro.Observable[V]) ro.Observable[V]) []int8 {
	t.Helper()

	vectors := VectorizeInt8[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int8, int8](
			vectors,
			ro.Map(func(v V) []int8 {
				var buffer [maxLanes]int8
				n := v.StorePart(buffer[:])

				return append([]int8{}, buffer[:n]...)
			}),
			ro.Flatten[int8](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt8 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt8(mask simd.Mask8s) []bool {
	lanes := lanesInt8()

	var buf [maxLanes]int8
	mask.ToInt8s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTrip(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		got := collectInt8Values[PartialInt8s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapes(t *testing.T) {
	t.Parallel()

	lanes := lanesInt8()

	// 35 items at 16 lanes is two full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeInt8[PartialInt8s](ro.FromSlice(rampInt8(35))))
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

func TestAddInt8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int8) int8 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectInt8Values[PartialInt8s](t, input, AddInt8(BroadcastInt8(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubInt8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int8) int8 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectInt8Values[PartialInt8s](t, input, SubInt8(BroadcastInt8(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestClampInt8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[int8](10, 50)),
		)
		assert.NoError(t, err)

		got := collectInt8Values[PartialInt8s](t, input, ClampInt8(BroadcastInt8(10), BroadcastInt8(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampInt8InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []int8{1, 30, 80}

	got := collectInt8Values[PartialInt8s](t, input, ClampInt8(BroadcastInt8(50), BroadcastInt8(10)))

	assert.Equal(t, []int8{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[int8](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsInt8ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialInt8s{}.LoadPart([]int8{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastInt8(7)), BroadcastInt8(0))

	assert.Equal(t, []int8{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsInt8MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialInt8s{}.LoadPart([]int8{0, 1, 0}, 3)

	lanes := maskLanesInt8(vector.Contains(BroadcastInt8(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmitted(t *testing.T) {
	t.Parallel()

	input := rampInt8(3)

	got := collectInt8Values[PartialInt8s](t, input, AddInt8(BroadcastInt8(100)))

	assert.Equal(t, []int8{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperations(t *testing.T) {
	t.Parallel()

	lanes := lanesInt8()

	tail := PartialInt8s{}.LoadPart([]int8{1, 2, 3}, 3).Add(BroadcastInt8(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, int8(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumInt8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[int8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, int8](ro.FromSlice(input), VectorizeInt8, ReduceSumInt8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinInt8MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[int8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, int8](ro.FromSlice(input), VectorizeInt8, ReduceMinInt8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxInt8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampInt8(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[int8]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, int8](ro.FromSlice(input), VectorizeInt8, ReduceMaxInt8),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsInt8ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[int8, PartialInt8s, bool](
			ro.FromSlice([]int8{1, 2, 3}),
			VectorizeInt8,
			ReduceContainsInt8(BroadcastInt8(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsInt8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		for _, target := range []int8{1, 3, 100, 127} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[int8, PartialInt8s, bool](
					ro.FromSlice(input),
					VectorizeInt8,
					ReduceContainsInt8(BroadcastInt8(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithInt8ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt8[PartialInt8s](ro.FromSlice(rampInt8(20)))
	right := VectorizeInt8[PartialInt8s](ro.FromSlice(rampInt8(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt8s, []int8, int8](
			AddWithInt8(right)(left),
			ro.Map(func(v PartialInt8s) []int8 { return v.Values() }),
			ro.Flatten[int8](),
		),
	)
	assert.NoError(t, err)

	want := make([]int8, 20)
	for i, v := range rampInt8(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMap(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int8, PartialInt8s, []int8, int8](
			ro.FromSlice(rampInt8(10)),
			VectorizeInt8,
			ro.Map(func(v PartialInt8s) []int8 {
				return v.Add(BroadcastInt8(42)).Min(BroadcastInt8(50)).Values()
			}),
			ro.Flatten[int8](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int8{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectors(t *testing.T) {
	t.Parallel()

	lanes := lanesInt8()
	batch := rampInt8(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt8s(batch)), AddInt8(simd.BroadcastInt8s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]int8
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSources(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[int8, PartialInt8s, int8](ro.Empty[int8](), VectorizeInt8, ReduceSumInt8),
	)
	assert.NoError(t, err)
	assert.Equal(t, []int8{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeInt8[PartialInt8s](ro.Empty[int8]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialInt8s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeInt8[PartialInt8s](ro.Throw[int8](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[int8, PartialInt8s, int8](ro.Throw[int8](assert.AnError), VectorizeInt8, ReduceSumInt8),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
