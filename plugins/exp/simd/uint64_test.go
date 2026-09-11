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
	"math"
	"simd"
	"slices"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func lanesUint64() int { return simd.BroadcastUint64s(0).Len() }

// sizeSweepUint64 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong. With only 2 uint64 lanes in a
// 128-bit register the entries overlap; duplicates are harmless.
func sizeSweepUint64() []int {
	lanes := lanesUint64()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint64 builds 1, 2, 3... uint64 has range to spare, so no wrapping is needed.
func rampUint64(size int) []uint64 {
	out := make([]uint64, size)
	for i := range out {
		out[i] = uint64(i + 1)
	}

	return out
}

// collectUint64Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint64s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint64Values[V Uint64Buffer[V]](t *testing.T, input []uint64, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint64 {
	t.Helper()

	vectors := VectorizeUint64[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint64, uint64](
			vectors,
			ro.Map(func(v V) []uint64 {
				var buffer [maxLanes]uint64
				n := v.StorePart(buffer[:])

				return append([]uint64{}, buffer[:n]...)
			}),
			ro.Flatten[uint64](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint64 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint64(mask simd.Mask64s) []bool {
	lanes := lanesUint64()

	var buf [maxLanes]int64
	mask.ToInt64s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripUint64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		got := collectUint64Values[PartialUint64s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint64(t *testing.T) {
	t.Parallel()

	lanes := lanesUint64()

	// 35 is 5*7, and uint64 lane counts are powers of two, so the sweep always ends
	// in a short tail — 17 full vectors plus a 1-lane tail at 2 lanes.
	vectors, err := ro.Collect(VectorizeUint64[PartialUint64s](ro.FromSlice(rampUint64(35))))
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

func TestAddUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint64) uint64 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectUint64Values[PartialUint64s](t, input, AddUint64(BroadcastUint64(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint64) uint64 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectUint64Values[PartialUint64s](t, input, SubUint64(BroadcastUint64(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Min and Max are synthesized from Less and IfElse because simd.Uint64s has neither
// method, so they are checked differentially against Go's builtins rather than
// trusted by construction.
func TestMinMaxUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		wantMin, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint64) uint64 { return min(v, 20) })),
		)
		assert.NoError(t, err)

		gotMin := collectUint64Values[PartialUint64s](t, input, MinUint64(BroadcastUint64(20)))

		assert.Equal(t, wantMin, gotMin, "min size %d", size)

		wantMax, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint64) uint64 { return max(v, 20) })),
		)
		assert.NoError(t, err)

		gotMax := collectUint64Values[PartialUint64s](t, input, MaxUint64(BroadcastUint64(20)))

		assert.Equal(t, wantMax, gotMax, "max size %d", size)
	}
}

func TestClampUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[uint64](10, 50)),
		)
		assert.NoError(t, err)

		got := collectUint64Values[PartialUint64s](t, input, ClampUint64(BroadcastUint64(10), BroadcastUint64(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// A single vector holding one lane below the lower bound and one above the upper
// exercises both of Clamp's selections at once — the shape where a synthesized
// Min or Max with swapped IfElse operands would corrupt exactly one of the two.
func TestClampUint64MethodBoundsBothWays(t *testing.T) {
	t.Parallel()

	lanes := lanesUint64()

	input := make([]uint64, lanes)
	input[0] = 1  // below lower
	input[1] = 80 // above upper
	for i := 2; i < lanes; i++ {
		input[i] = 30 // in range
	}

	got := PartialUint64s{}.Load(input).Clamp(BroadcastUint64(10), BroadcastUint64(50))

	want := make([]uint64, lanes)
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
func TestClampUint64InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []uint64{1, 30, 80}

	got := collectUint64Values[PartialUint64s](t, input, ClampUint64(BroadcastUint64(50), BroadcastUint64(10)))

	assert.Equal(t, []uint64{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[uint64](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
//
// A 128-bit register holds only 2 uint64 lanes, so the fixture is built to lanes-1
// rather than a hardcoded length that could exceed the width.
func TestContainsUint64ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	valid := lanesUint64() - 1

	input := make([]uint64, valid)
	for i := range input {
		if i%2 == 0 {
			input[i] = 7
		} else {
			input[i] = uint64(i) + 100 // never equal to the target
		}
	}

	vector := PartialUint64s{}.LoadPart(input, valid)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint64(7)), BroadcastUint64(0))

	want := make([]uint64, valid)
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
func TestContainsUint64MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	valid := lanesUint64() - 1

	input := make([]uint64, valid) // zeros at even indexes, non-zero elsewhere
	for i := range input {
		if i%2 == 1 {
			input[i] = uint64(i)
		}
	}

	vector := PartialUint64s{}.LoadPart(input, valid)

	lanes := maskLanesUint64(vector.Contains(BroadcastUint64(0)))

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
func TestPaddedLanesNeverEmittedUint64(t *testing.T) {
	t.Parallel()

	input := rampUint64(lanesUint64() - 1)

	got := collectUint64Values[PartialUint64s](t, input, AddUint64(BroadcastUint64(100)))

	want := make([]uint64, len(input))
	for i, v := range input {
		want[i] = v + 100
	}

	assert.Equal(t, want, got)
	assert.Len(t, got, len(input), "a short input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint64(t *testing.T) {
	t.Parallel()

	lanes := lanesUint64()
	valid := lanes - 1

	tail := PartialUint64s{}.LoadPart(rampUint64(valid), valid).Add(BroadcastUint64(100))

	assert.Equal(t, valid, tail.Count())

	for lane := valid; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range valid {
		assert.Equal(t, uint64(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

func TestReduceSumUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, ReduceSumUint64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinUint64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, ReduceMinUint64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxUint64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, ReduceMaxUint64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail. A lanes-1
// input guarantees the final vector is short.
func TestReduceContainsUint64ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint64, PartialUint64s, bool](
			ro.FromSlice(rampUint64(lanesUint64()-1)),
			VectorizeUint64,
			ReduceContainsUint64(BroadcastUint64(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		// The ramp is 1..size, so 35 is present at exactly the largest swept size
		// and absent at every other, exercising both outcomes.
		for _, target := range []uint64{1, 3, 35, 100} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint64, PartialUint64s, bool](
					ro.FromSlice(input),
					VectorizeUint64,
					ReduceContainsUint64(BroadcastUint64(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

func TestAddWithUint64ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint64[PartialUint64s](ro.FromSlice(rampUint64(20)))
	right := VectorizeUint64[PartialUint64s](ro.FromSlice(rampUint64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint64s, []uint64, uint64](
			AddWithUint64(right)(left),
			ro.Map(func(v PartialUint64s) []uint64 { return v.Values() }),
			ro.Flatten[uint64](),
		),
	)
	assert.NoError(t, err)

	want := make([]uint64, 20)
	for i, v := range rampUint64(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint64, PartialUint64s, []uint64, uint64](
			ro.FromSlice(rampUint64(10)),
			VectorizeUint64,
			ro.Map(func(v PartialUint64s) []uint64 {
				return v.Add(BroadcastUint64(42)).Min(BroadcastUint64(50)).Values()
			}),
			ro.Flatten[uint64](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint64{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// There is no uint64 counterpart to TestOperatorsAcceptStdlibVectors: simd.Uint64s
// lacks Min and Max, so it cannot satisfy Uint64Vector. Only PartialUint64s — which
// synthesizes both from Less and IfElse — flows through these operators.

func TestEmptyAndErrorSourcesUint64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint64, PartialUint64s, uint64](ro.Empty[uint64](), VectorizeUint64, ReduceSumUint64),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint64{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint64[PartialUint64s](ro.Empty[uint64]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint64s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint64[PartialUint64s](ro.Throw[uint64](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint64, PartialUint64s, uint64](ro.Throw[uint64](assert.AnError), VectorizeUint64, ReduceSumUint64),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

// Min and Max are synthesized from Less, so this is the test that the unsigned Less is
// the one being used. math.MaxUint64 read as int64 is -1: under a signed comparison it
// would come out as the smallest value here rather than the largest.
func TestMinMaxUint64TreatsHighBitAsLarge(t *testing.T) {
	t.Parallel()

	input := []uint64{100, math.MaxUint64, 1000}

	capped := collectUint64Values[PartialUint64s](t, input, MinUint64(BroadcastUint64(1<<63)))
	assert.Equal(t, []uint64{100, 1 << 63, 1000}, capped,
		"math.MaxUint64 must be clamped down, not treated as smaller than 100")

	raised := collectUint64Values[PartialUint64s](t, input, MaxUint64(BroadcastUint64(500)))
	assert.Equal(t, []uint64{500, math.MaxUint64, 1000}, raised)

	largest, err := ro.Collect(
		ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, ReduceMaxUint64),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint64{math.MaxUint64}, largest)

	smallest, err := ro.Collect(
		ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, ReduceMinUint64),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint64{100}, smallest)
}
