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

func lanesFloat32() int { return simd.BroadcastFloat32s(0).Len() }

// sizeSweepFloat32 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepFloat32() []int {
	lanes := lanesFloat32()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampFloat32 builds 1.5, 2.5, 3.5... wrapping at 1000. The half is deliberate: every
// value stays exactly representable, so the differential tests can assert equality
// rather than a tolerance, while still exercising a non-integral mantissa.
func rampFloat32(size int) []float32 {
	out := make([]float32, size)
	for i := range out {
		out[i] = float32(i%1000+1) + 0.5
	}

	return out
}

// isNaNFloat32 tests for NaN without math.IsNaN, whose float32 parameter does not
// survive into the float32 twin of this file.
func isNaNFloat32(value float32) bool { return value != value }

// assertLanesEqualFloat32 compares lane by lane, counting NaN as equal to NaN. Plain
// equality cannot: NaN matches nothing, including itself.
func assertLanesEqualFloat32(t *testing.T, want, got []float32, context string) {
	t.Helper()

	assert.Len(t, got, len(want), context)

	for i := range want {
		if isNaNFloat32(want[i]) {
			assert.True(t, isNaNFloat32(got[i]), "%s: lane %d must be NaN, got %v", context, i, got[i])

			continue
		}

		assert.Equal(t, want[i], got[i], "%s: lane %d", context, i)
	}
}

// collectFloat32Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialFloat32s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails to
// compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectFloat32Values[V Float32Buffer[V]](t *testing.T, input []float32, operators ...func(ro.Observable[V]) ro.Observable[V]) []float32 {
	t.Helper()

	vectors := VectorizeFloat32[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []float32, float32](
			vectors,
			ro.Map(func(v V) []float32 {
				var buffer [maxLanes]float32
				n := v.StorePart(buffer[:])

				return append([]float32{}, buffer[:n]...)
			}),
			ro.Flatten[float32](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesFloat32 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
//
// The buffer is an integer one because a mask converts only to the integer vector type
// of its width.
func maskLanesFloat32(mask simd.Mask32s) []bool {
	lanes := lanesFloat32()

	var buf [maxLanes]int32
	mask.ToInt32s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripFloat32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		got := collectFloat32Values[PartialFloat32s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesFloat32(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()

	vectors, err := ro.Collect(VectorizeFloat32[PartialFloat32s](ro.FromSlice(rampFloat32(35))))
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

func TestAddFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float32) float32 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectFloat32Values[PartialFloat32s](t, input, AddFloat32(BroadcastFloat32(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float32) float32 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectFloat32Values[PartialFloat32s](t, input, SubFloat32(BroadcastFloat32(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestMulFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float32) float32 { return v * 3 })),
		)
		assert.NoError(t, err)

		got := collectFloat32Values[PartialFloat32s](t, input, MulFloat32(BroadcastFloat32(3)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Div exists for the float types alone, so it has no integer counterpart to be ported
// from.
func TestDivFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float32) float32 { return v / 4 })),
		)
		assert.NoError(t, err)

		got := collectFloat32Values[PartialFloat32s](t, input, DivFloat32(BroadcastFloat32(4)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Dividing by zero is not an error in Go and must not become one here.
func TestDivFloat32ByZeroFollowsGo(t *testing.T) {
	t.Parallel()

	zero := float32(0)
	want := []float32{1 / zero, -1 / zero, zero / zero}

	got := collectFloat32Values[PartialFloat32s](t, []float32{1, -1, 0}, DivFloat32(BroadcastFloat32(0)))

	assertLanesEqualFloat32(t, want, got, "division by zero")
}

// Padded lanes are zero-filled, so an unmasked Div of two short vectors would compute
// 0/0 and leave NaN behind — which a later Min or Max would then propagate into a
// valid lane.
func TestDivWithFloat32LeavesNoNaNInPadding(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()

	left := VectorizeFloat32[PartialFloat32s](ro.FromSlice([]float32{1}))
	right := VectorizeFloat32[PartialFloat32s](ro.FromSlice([]float32{2}))

	vectors, err := ro.Collect(DivWithFloat32(right)(left))
	assert.NoError(t, err)
	assert.Len(t, vectors, 1)

	assert.Equal(t, []float32{0.5}, vectors[0].Values())

	for lane := 1; lane < lanes; lane++ {
		assert.False(t, isNaNFloat32(vectors[0].rawLane(lane)),
			"padded lane %d must not hold the NaN that 0/0 produces", lane)
		assert.Zero(t, vectors[0].rawLane(lane), "padded lane %d", lane)
	}
}

func TestClampFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[float32](10, 50)),
		)
		assert.NoError(t, err)

		got := collectFloat32Values[PartialFloat32s](t, input, ClampFloat32(BroadcastFloat32(10), BroadcastFloat32(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampFloat32InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []float32{1, 30, 80}

	got := collectFloat32Values[PartialFloat32s](t, input, ClampFloat32(BroadcastFloat32(50), BroadcastFloat32(10)))

	assert.Equal(t, []float32{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[float32](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsFloat32ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()

	input := make([]float32, lanes)
	for i := range input {
		input[i] = 1
	}
	input[0] = 7

	vector := PartialFloat32s{}.Load(input)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastFloat32(7)), BroadcastFloat32(0))

	want := make([]float32, lanes)
	want[0] = 7

	assert.Equal(t, want, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match — even
// when searching for the very zero that padding is filled with.
func TestContainsFloat32MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	// A single valid lane is the only tail shape that leaves padding behind at every
	// supported vector width, float32's two-lane minimum included.
	vector := PartialFloat32s{}.LoadPart([]float32{0}, 1)

	lanes := maskLanesFloat32(vector.Contains(BroadcastFloat32(0)))

	assert.True(t, lanes[0], "the valid lane equals zero and must match")

	for lane := 1; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// NaN equals nothing, itself included, so searching for it finds nothing. Go's own ==
// behaves the same way, so a scalar search would report the same.
func TestContainsFloat32NaNNeverMatches(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())

	vector := PartialFloat32s{}.LoadPart([]float32{nan}, 1)

	assert.False(t, maskLanesFloat32(vector.Contains(BroadcastFloat32(nan)))[0],
		"a NaN lane must not match a NaN target")

	found, err := ro.Collect(
		ro.Pipe2[float32, PartialFloat32s, bool](
			ro.FromSlice([]float32{1, nan, 3}),
			VectorizeFloat32,
			ReduceContainsFloat32(BroadcastFloat32(nan)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, found)
}

// Padded lanes must not leak into the output. Adding to a short tail is where a missing
// mask would show up as extra values.
func TestPaddedLanesNeverEmittedFloat32(t *testing.T) {
	t.Parallel()

	input := rampFloat32(lanesFloat32() + 1)

	got := collectFloat32Values[PartialFloat32s](t, input, AddFloat32(BroadcastFloat32(100)))

	want := make([]float32, len(input))
	for i, v := range input {
		want[i] = v + 100
	}

	assert.Equal(t, want, got)
	assert.Len(t, got, len(input), "the tail must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsFloat32(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()

	tail := PartialFloat32s{}.LoadPart([]float32{1}, 1).Add(BroadcastFloat32(100))

	assert.Equal(t, 1, tail.Count())
	assert.Equal(t, float32(101), tail.rawLane(0), "the valid lane")

	for lane := 1; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}
}

// Hardware disagrees about NaN in Min and Max — x86 discards it, arm64 keeps it — so
// the Partial types force it in rather than inheriting the machine's answer. Go's own
// min and max builtins are the oracle.
func TestMinMaxFloat32PropagateNaN(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())

	for _, pair := range [][2]float32{{1, 2}, {nan, 2}, {1, nan}, {nan, nan}} {
		left := PartialFloat32s{}.LoadPart([]float32{pair[0]}, 1)
		right := BroadcastFloat32(pair[1])

		assertLanesEqualFloat32(t,
			[]float32{min(pair[0], pair[1])}, left.Min(right).Values(),
			"Min must agree with the builtin")

		assertLanesEqualFloat32(t,
			[]float32{max(pair[0], pair[1])}, left.Max(right).Values(),
			"Max must agree with the builtin")
	}
}

// restoreNaN is what makes Min and Max architecture-independent, but the difference is
// invisible through Min and Max on hardware that already propagates NaN — arm64 does,
// so on arm64 removing the call entirely still passes the test above. Feeding restoreNaN
// the result x86 would have produced, with the NaN discarded, exercises it everywhere.
func TestRestoreNaNFloat32PutsNaNBackWhicheverOperandHeldIt(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())
	discarded := simd.BroadcastFloat32s(2)

	left := PartialFloat32s{}.LoadPart([]float32{nan}, 1)
	fromLeft := left.apply(left.restoreNaN(discarded, BroadcastFloat32(2).vec))
	assert.True(t, isNaNFloat32(fromLeft.Values()[0]), "a NaN receiver must survive")

	right := PartialFloat32s{}.LoadPart([]float32{2}, 1)
	fromRight := right.apply(right.restoreNaN(discarded, BroadcastFloat32(nan).vec))
	assert.True(t, isNaNFloat32(fromRight.Values()[0]), "a NaN operand must survive")

	// A lane where neither operand is NaN must keep the computed result untouched.
	clean := right.apply(right.restoreNaN(discarded, BroadcastFloat32(2).vec))
	assert.Equal(t, float32(2), clean.Values()[0])
}

// Reductions go the opposite way from the element-wise methods: they compare with < and
// >, both false for NaN, so a NaN never displaces the accumulator. That is what ro.Min
// and ro.Max do, and agreeing with them is the whole point.
func TestReduceMinMaxFloat32MatchCoreOnNaN(t *testing.T) {
	t.Parallel()

	nan := float32(math.NaN())

	for _, input := range [][]float32{
		{3, nan, 1, 5},
		{nan, 1, 2},
		{1, 2, nan},
		{nan},
	} {
		wantMin, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[float32]()))
		assert.NoError(t, err)

		gotMin, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, ReduceMinFloat32),
		)
		assert.NoError(t, err)
		assertLanesEqualFloat32(t, wantMin, gotMin, "ReduceMin with NaN")

		wantMax, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[float32]()))
		assert.NoError(t, err)

		gotMax, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, ReduceMaxFloat32),
		)
		assert.NoError(t, err)
		assertLanesEqualFloat32(t, wantMax, gotMax, "ReduceMax with NaN")
	}
}

func TestReduceSumFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[float32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, ReduceSumFloat32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinFloat32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[float32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, ReduceMinFloat32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxFloat32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampFloat32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[float32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, ReduceMaxFloat32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded lanes,
// so an unmasked compare would report a match on any short tail.
func TestReduceContainsFloat32ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[float32, PartialFloat32s, bool](
			ro.FromSlice([]float32{1, 2, 3}),
			VectorizeFloat32,
			ReduceContainsFloat32(BroadcastFloat32(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsFloat32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		for _, target := range []float32{1.5, 3.5, 100.5, 9999.5} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[float32, PartialFloat32s, bool](
					ro.FromSlice(input),
					VectorizeFloat32,
					ReduceContainsFloat32(BroadcastFloat32(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %v", size, target)
		}
	}
}

func TestAddWithFloat32ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeFloat32[PartialFloat32s](ro.FromSlice(rampFloat32(20)))
	right := VectorizeFloat32[PartialFloat32s](ro.FromSlice(rampFloat32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialFloat32s, []float32, float32](
			AddWithFloat32(right)(left),
			ro.Map(func(v PartialFloat32s) []float32 { return v.Values() }),
			ro.Flatten[float32](),
		),
	)
	assert.NoError(t, err)

	want := make([]float32, 20)
	for i, v := range rampFloat32(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapFloat32(t *testing.T) {
	t.Parallel()

	input := rampFloat32(10)

	values, err := ro.Collect(
		ro.Pipe3[float32, PartialFloat32s, []float32, float32](
			ro.FromSlice(input),
			VectorizeFloat32,
			ro.Map(func(v PartialFloat32s) []float32 {
				return v.Add(BroadcastFloat32(42)).Min(BroadcastFloat32(50)).Values()
			}),
			ro.Flatten[float32](),
		),
	)
	assert.NoError(t, err)

	want := make([]float32, len(input))
	for i, v := range input {
		want[i] = min(v+42, 50)
	}

	assert.Equal(t, want, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsFloat32(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()
	batch := rampFloat32(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadFloat32s(batch)), AddFloat32(simd.BroadcastFloat32s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]float32
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesFloat32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[float32, PartialFloat32s, float32](ro.Empty[float32](), VectorizeFloat32, ReduceSumFloat32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []float32{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeFloat32[PartialFloat32s](ro.Empty[float32]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialFloat32s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeFloat32[PartialFloat32s](ro.Throw[float32](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[float32, PartialFloat32s, float32](ro.Throw[float32](assert.AnError), VectorizeFloat32, ReduceSumFloat32),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
