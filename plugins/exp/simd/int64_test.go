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

func lanesInt64() int { return simd.BroadcastInt64s(0).Len() }

// sizeSweepInt64 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong. With only 2 int64 lanes in a
// 128-bit register the entries overlap; duplicates are harmless.
func sizeSweepInt64() []int {
	lanes := lanesInt64()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt64 builds 1, 2, 3... int64 has range to spare, so no wrapping is needed.
func rampInt64(size int) []int64 {
	out := make([]int64, size)
	for i := range out {
		out[i] = int64(i + 1)
	}

	return out
}

// collectInt64Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt64s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt64Values[V Int64Buffer[V]](t *testing.T, input []int64, operators ...func(ro.Observable[V]) ro.Observable[V]) []int64 {
	t.Helper()

	vectors := VectorizeInt64[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int64, int64](
			vectors,
			ro.Map(func(v V) []int64 {
				var buffer [maxLanes]int64
				n := v.StorePart(buffer[:])

				return append([]int64{}, buffer[:n]...)
			}),
			ro.Flatten[int64](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt64 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt64(mask simd.Mask64s) []bool {
	lanes := lanesInt64()

	var buf [maxLanes]int64
	mask.ToInt64s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripInt64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		got := collectInt64Values[PartialInt64s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesInt64(t *testing.T) {
	t.Parallel()

	lanes := lanesInt64()

	// 35 is 5*7, and int64 lane counts are powers of two, so the sweep always ends
	// in a short tail — 17 full vectors plus a 1-lane tail at 2 lanes.
	vectors, err := ro.Collect(VectorizeInt64[PartialInt64s](ro.FromSlice(rampInt64(35))))
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

func TestAddInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int64) int64 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectInt64Values[PartialInt64s](t, input, AddInt64(BroadcastInt64(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int64) int64 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectInt64Values[PartialInt64s](t, input, SubInt64(BroadcastInt64(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Min and Max are synthesized from Less and IfElse because simd.Int64s has neither
// method, so they are checked differentially against Go's builtins rather than
// trusted by construction.
func TestMinMaxInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		wantMin, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int64) int64 { return min(v, 20) })),
		)
		assert.NoError(t, err)

		gotMin := collectInt64Values[PartialInt64s](t, input, MinInt64(BroadcastInt64(20)))

		assert.Equal(t, wantMin, gotMin, "min size %d", size)

		wantMax, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v int64) int64 { return max(v, 20) })),
		)
		assert.NoError(t, err)

		gotMax := collectInt64Values[PartialInt64s](t, input, MaxInt64(BroadcastInt64(20)))

		assert.Equal(t, wantMax, gotMax, "max size %d", size)
	}
}

func TestClampInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[int64](10, 50)),
		)
		assert.NoError(t, err)

		got := collectInt64Values[PartialInt64s](t, input, ClampInt64(BroadcastInt64(10), BroadcastInt64(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// A single vector holding one lane below the lower bound and one above the upper
// exercises both of Clamp's selections at once — the shape where a synthesized
// Min or Max with swapped IfElse operands would corrupt exactly one of the two.
func TestClampInt64MethodBoundsBothWays(t *testing.T) {
	t.Parallel()

	lanes := lanesInt64()

	input := make([]int64, lanes)
	input[0] = 1  // below lower
	input[1] = 80 // above upper
	for i := 2; i < lanes; i++ {
		input[i] = 30 // in range
	}

	got := PartialInt64s{}.Load(input).Clamp(BroadcastInt64(10), BroadcastInt64(50))

	want := make([]int64, lanes)
	want[0] = 10
	want[1] = 50
	for i := 2; i < lanes; i++ {
		want[i] = 30
	}

	assert.Equal(t, want, got.Values())
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampInt64InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []int64{1, 30, 80}

	got := collectInt64Values[PartialInt64s](t, input, ClampInt64(BroadcastInt64(50), BroadcastInt64(10)))

	assert.Equal(t, []int64{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[int64](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
//
// A 128-bit register holds only 2 int64 lanes, so the fixture is built to lanes-1
// rather than a hardcoded length that could exceed the width.
func TestContainsInt64ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	valid := lanesInt64() - 1

	input := make([]int64, valid)
	for i := range input {
		if i%2 == 0 {
			input[i] = 7
		} else {
			input[i] = int64(i) + 100 // never equal to the target
		}
	}

	vector := PartialInt64s{}.LoadPart(input, valid)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastInt64(7)), BroadcastInt64(0))

	want := make([]int64, valid)
	for i := range input {
		if input[i] == 7 {
			want[i] = 7
		}
	}

	assert.Equal(t, want, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsInt64MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	valid := lanesInt64() - 1

	input := make([]int64, valid) // zeros at even indexes, non-zero elsewhere
	for i := range input {
		if i%2 == 1 {
			input[i] = int64(i)
		}
	}

	vector := PartialInt64s{}.LoadPart(input, valid)

	lanes := maskLanesInt64(vector.Contains(BroadcastInt64(0)))

	for i := range valid {
		assert.Equal(t, input[i] == 0, lanes[i], "valid lane %d", i)
	}

	for lane := valid; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedInt64(t *testing.T) {
	t.Parallel()

	input := rampInt64(lanesInt64() - 1)

	got := collectInt64Values[PartialInt64s](t, input, AddInt64(BroadcastInt64(100)))

	want := make([]int64, len(input))
	for i, v := range input {
		want[i] = v + 100
	}

	assert.Equal(t, want, got)
	assert.Len(t, got, len(input), "a short input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsInt64(t *testing.T) {
	t.Parallel()

	lanes := lanesInt64()
	valid := lanes - 1

	tail := PartialInt64s{}.LoadPart(rampInt64(valid), valid).Add(BroadcastInt64(100))

	assert.Equal(t, valid, tail.Count())

	for lane := valid; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range valid {
		assert.Equal(t, int64(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[int64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, int64](ro.FromSlice(input), VectorizeInt64, ReduceSumInt64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinInt64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[int64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, int64](ro.FromSlice(input), VectorizeInt64, ReduceMinInt64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxInt64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampInt64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[int64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, int64](ro.FromSlice(input), VectorizeInt64, ReduceMaxInt64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail. A lanes-1
// input guarantees the final vector is short.
func TestReduceContainsInt64ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[int64, PartialInt64s, bool](
			ro.FromSlice(rampInt64(lanesInt64()-1)),
			VectorizeInt64,
			ReduceContainsInt64(BroadcastInt64(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsInt64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		// The ramp is 1..size, so 35 is present at exactly the largest swept size
		// and absent at every other, exercising both outcomes.
		for _, target := range []int64{1, 3, 35, 100} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[int64, PartialInt64s, bool](
					ro.FromSlice(input),
					VectorizeInt64,
					ReduceContainsInt64(BroadcastInt64(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithInt64ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt64[PartialInt64s](ro.FromSlice(rampInt64(20)))
	right := VectorizeInt64[PartialInt64s](ro.FromSlice(rampInt64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt64s, []int64, int64](
			AddWithInt64(right)(left),
			ro.Map(func(v PartialInt64s) []int64 { return v.Values() }),
			ro.Flatten[int64](),
		),
	)
	assert.NoError(t, err)

	want := make([]int64, 20)
	for i, v := range rampInt64(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int64, PartialInt64s, []int64, int64](
			ro.FromSlice(rampInt64(10)),
			VectorizeInt64,
			ro.Map(func(v PartialInt64s) []int64 {
				return v.Add(BroadcastInt64(42)).Min(BroadcastInt64(50)).Values()
			}),
			ro.Flatten[int64](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int64{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// There is no int64 counterpart to TestOperatorsAcceptStdlibVectors: simd.Int64s
// lacks Min and Max, so it cannot satisfy Int64Vector. Only PartialInt64s — which
// synthesizes both from Less and IfElse — flows through these operators.

func TestEmptyAndErrorSourcesInt64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[int64, PartialInt64s, int64](ro.Empty[int64](), VectorizeInt64, ReduceSumInt64),
	)
	assert.NoError(t, err)
	assert.Equal(t, []int64{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeInt64[PartialInt64s](ro.Empty[int64]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialInt64s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeInt64[PartialInt64s](ro.Throw[int64](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[int64, PartialInt64s, int64](ro.Throw[int64](assert.AnError), VectorizeInt64, ReduceSumInt64),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
