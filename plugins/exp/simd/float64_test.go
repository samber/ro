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

func lanesFloat64() int { return simd.BroadcastFloat64s(0).Len() }

// sizeSweepFloat64 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepFloat64() []int {
	lanes := lanesFloat64()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampFloat64 builds 1.5, 2.5, 3.5... wrapping at 1000. The half is deliberate: every
// value stays exactly representable, so the differential tests can assert equality
// rather than a tolerance, while still exercising a non-integral mantissa.
func rampFloat64(size int) []float64 {
	out := make([]float64, size)
	for i := range out {
		out[i] = float64(i%1000+1) + 0.5
	}

	return out
}

// isNaNFloat64 tests for NaN without math.IsNaN, whose float64 parameter does not
// survive into the float32 twin of this file.
func isNaNFloat64(value float64) bool { return value != value }

// assertLanesEqualFloat64 compares lane by lane, counting NaN as equal to NaN. Plain
// equality cannot: NaN matches nothing, including itself.
func assertLanesEqualFloat64(t *testing.T, want, got []float64, context string) {
	t.Helper()

	assert.Len(t, got, len(want), context)

	for i := range want {
		if isNaNFloat64(want[i]) {
			assert.True(t, isNaNFloat64(got[i]), "%s: lane %d must be NaN, got %v", context, i, got[i])

			continue
		}

		assert.Equal(t, want[i], got[i], "%s: lane %d", context, i)
	}
}

// collectFloat64Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialFloat64s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails to
// compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectFloat64Values[V Float64Buffer[V]](t *testing.T, input []float64, operators ...func(ro.Observable[V]) ro.Observable[V]) []float64 {
	t.Helper()

	vectors := VectorizeFloat64[V](ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []float64, float64](
			vectors,
			ro.Map(func(v V) []float64 {
				var buffer [maxLanes]float64
				n := v.StorePart(buffer[:])

				return append([]float64{}, buffer[:n]...)
			}),
			ro.Flatten[float64](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesFloat64 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
//
// The buffer is an integer one because a mask converts only to the integer vector type
// of its width.
func maskLanesFloat64(mask simd.Mask64s) []bool {
	lanes := lanesFloat64()

	var buf [maxLanes]int64
	mask.ToInt64s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func TestVectorizeRoundTripFloat64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		got := collectFloat64Values[PartialFloat64s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesFloat64(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()

	vectors, err := ro.Collect(VectorizeFloat64[PartialFloat64s](ro.FromSlice(rampFloat64(35))))
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

func TestAddFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float64) float64 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectFloat64Values[PartialFloat64s](t, input, AddFloat64(BroadcastFloat64(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestSubFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float64) float64 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectFloat64Values[PartialFloat64s](t, input, SubFloat64(BroadcastFloat64(7)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestMulFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float64) float64 { return v * 3 })),
		)
		assert.NoError(t, err)

		got := collectFloat64Values[PartialFloat64s](t, input, MulFloat64(BroadcastFloat64(3)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Div exists for the float types alone, so it has no integer counterpart to be ported
// from.
func TestDivFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v float64) float64 { return v / 4 })),
		)
		assert.NoError(t, err)

		got := collectFloat64Values[PartialFloat64s](t, input, DivFloat64(BroadcastFloat64(4)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Dividing by zero is not an error in Go and must not become one here.
func TestDivFloat64ByZeroFollowsGo(t *testing.T) {
	t.Parallel()

	zero := float64(0)
	want := []float64{1 / zero, -1 / zero, zero / zero}

	got := collectFloat64Values[PartialFloat64s](t, []float64{1, -1, 0}, DivFloat64(BroadcastFloat64(0)))

	assertLanesEqualFloat64(t, want, got, "division by zero")
}

// Padded lanes are zero-filled, so an unmasked Div of two short vectors would compute
// 0/0 and leave NaN behind — which a later Min or Max would then propagate into a
// valid lane.
func TestDivWithFloat64LeavesNoNaNInPadding(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()

	left := VectorizeFloat64[PartialFloat64s](ro.FromSlice([]float64{1}))
	right := VectorizeFloat64[PartialFloat64s](ro.FromSlice([]float64{2}))

	vectors, err := ro.Collect(DivWithFloat64(right)(left))
	assert.NoError(t, err)
	assert.Len(t, vectors, 1)

	assert.Equal(t, []float64{0.5}, vectors[0].Values())

	for lane := 1; lane < lanes; lane++ {
		assert.False(t, isNaNFloat64(vectors[0].rawLane(lane)),
			"padded lane %d must not hold the NaN that 0/0 produces", lane)
		assert.Zero(t, vectors[0].rawLane(lane), "padded lane %d", lane)
	}
}

func TestClampFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Clamp[float64](10, 50)),
		)
		assert.NoError(t, err)

		got := collectFloat64Values[PartialFloat64s](t, input, ClampFloat64(BroadcastFloat64(10), BroadcastFloat64(50)))

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Inverted bounds are a programmer error, but the vector operators cannot detect it:
// the bounds are opaque vectors, and core ro.Clamp's construction-time panic has no
// equivalent here. Pin down what actually happens so the godoc can describe it
// truthfully — Max(lower) then Min(upper) collapses every lane to upper.
func TestClampFloat64InvertedBoundsCollapseToUpper(t *testing.T) {
	t.Parallel()

	input := []float64{1, 30, 80}

	got := collectFloat64Values[PartialFloat64s](t, input, ClampFloat64(BroadcastFloat64(50), BroadcastFloat64(10)))

	assert.Equal(t, []float64{10, 10, 10}, got,
		"with lower > upper every lane becomes upper, regardless of its value")

	// Core ro.Clamp rejects the same bounds outright rather than producing a value.
	assert.PanicsWithError(t, "ro.Clamp: lower must be less than or equal to upper", func() {
		ro.Clamp[float64](50, 10)
	})
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsFloat64ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()

	input := make([]float64, lanes)
	for i := range input {
		input[i] = 1
	}
	input[0] = 7

	vector := PartialFloat64s{}.Load(input)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastFloat64(7)), BroadcastFloat64(0))

	want := make([]float64, lanes)
	want[0] = 7

	assert.Equal(t, want, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match — even
// when searching for the very zero that padding is filled with.
func TestContainsFloat64MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	// A single valid lane is the only tail shape that leaves padding behind at every
	// supported vector width, float64's two-lane minimum included.
	vector := PartialFloat64s{}.LoadPart([]float64{0}, 1)

	lanes := maskLanesFloat64(vector.Contains(BroadcastFloat64(0)))

	assert.True(t, lanes[0], "the valid lane equals zero and must match")

	for lane := 1; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// NaN equals nothing, itself included, so searching for it finds nothing. Go's own ==
// behaves the same way, so a scalar search would report the same.
func TestContainsFloat64NaNNeverMatches(t *testing.T) {
	t.Parallel()

	nan := float64(math.NaN())

	vector := PartialFloat64s{}.LoadPart([]float64{nan}, 1)

	assert.False(t, maskLanesFloat64(vector.Contains(BroadcastFloat64(nan)))[0],
		"a NaN lane must not match a NaN target")

	found, err := ro.Collect(
		ro.Pipe2[float64, PartialFloat64s, bool](
			ro.FromSlice([]float64{1, nan, 3}),
			VectorizeFloat64,
			ReduceContainsFloat64(BroadcastFloat64(nan)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, found)
}

// Padded lanes must not leak into the output. Adding to a short tail is where a missing
// mask would show up as extra values.
func TestPaddedLanesNeverEmittedFloat64(t *testing.T) {
	t.Parallel()

	input := rampFloat64(lanesFloat64() + 1)

	got := collectFloat64Values[PartialFloat64s](t, input, AddFloat64(BroadcastFloat64(100)))

	want := make([]float64, len(input))
	for i, v := range input {
		want[i] = v + 100
	}

	assert.Equal(t, want, got)
	assert.Len(t, got, len(input), "the tail must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsFloat64(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()

	tail := PartialFloat64s{}.LoadPart([]float64{1}, 1).Add(BroadcastFloat64(100))

	assert.Equal(t, 1, tail.Count())
	assert.Equal(t, float64(101), tail.rawLane(0), "the valid lane")

	for lane := 1; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}
}

// Hardware disagrees about NaN in Min and Max — x86 discards it, arm64 keeps it — so
// the Partial types force it in rather than inheriting the machine's answer. Go's own
// min and max builtins are the oracle.
func TestMinMaxFloat64PropagateNaN(t *testing.T) {
	t.Parallel()

	nan := float64(math.NaN())

	for _, pair := range [][2]float64{{1, 2}, {nan, 2}, {1, nan}, {nan, nan}} {
		left := PartialFloat64s{}.LoadPart([]float64{pair[0]}, 1)
		right := BroadcastFloat64(pair[1])

		assertLanesEqualFloat64(t,
			[]float64{min(pair[0], pair[1])}, left.Min(right).Values(),
			"Min must agree with the builtin")

		assertLanesEqualFloat64(t,
			[]float64{max(pair[0], pair[1])}, left.Max(right).Values(),
			"Max must agree with the builtin")
	}
}

// restoreNaN is what makes Min and Max architecture-independent, but the difference is
// invisible through Min and Max on hardware that already propagates NaN — arm64 does,
// so on arm64 removing the call entirely still passes the test above. Feeding restoreNaN
// the result x86 would have produced, with the NaN discarded, exercises it everywhere.
func TestRestoreNaNFloat64PutsNaNBackWhicheverOperandHeldIt(t *testing.T) {
	t.Parallel()

	nan := float64(math.NaN())
	discarded := simd.BroadcastFloat64s(2)

	left := PartialFloat64s{}.LoadPart([]float64{nan}, 1)
	fromLeft := left.apply(left.restoreNaN(discarded, BroadcastFloat64(2).vec))
	assert.True(t, isNaNFloat64(fromLeft.Values()[0]), "a NaN receiver must survive")

	right := PartialFloat64s{}.LoadPart([]float64{2}, 1)
	fromRight := right.apply(right.restoreNaN(discarded, BroadcastFloat64(nan).vec))
	assert.True(t, isNaNFloat64(fromRight.Values()[0]), "a NaN operand must survive")

	// A lane where neither operand is NaN must keep the computed result untouched.
	clean := right.apply(right.restoreNaN(discarded, BroadcastFloat64(2).vec))
	assert.Equal(t, float64(2), clean.Values()[0])
}

// Reductions go the opposite way from the element-wise methods: they compare with < and
// >, both false for NaN, so a NaN never displaces the accumulator. That is what ro.Min
// and ro.Max do, and agreeing with them is the whole point.
func TestReduceMinMaxFloat64MatchCoreOnNaN(t *testing.T) {
	t.Parallel()

	nan := float64(math.NaN())

	for _, input := range [][]float64{
		{3, nan, 1, 5},
		{nan, 1, 2},
		{1, 2, nan},
		{nan},
	} {
		wantMin, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[float64]()))
		assert.NoError(t, err)

		gotMin, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, ReduceMinFloat64),
		)
		assert.NoError(t, err)
		assertLanesEqualFloat64(t, wantMin, gotMin, "ReduceMin with NaN")

		wantMax, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[float64]()))
		assert.NoError(t, err)

		gotMax, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, ReduceMaxFloat64),
		)
		assert.NoError(t, err)
		assertLanesEqualFloat64(t, wantMax, gotMax, "ReduceMax with NaN")
	}
}

func TestReduceSumFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[float64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, ReduceSumFloat64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinFloat64MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[float64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, ReduceMinFloat64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxFloat64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampFloat64(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[float64]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, ReduceMaxFloat64),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded lanes,
// so an unmasked compare would report a match on any short tail.
func TestReduceContainsFloat64ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[float64, PartialFloat64s, bool](
			ro.FromSlice([]float64{1, 2, 3}),
			VectorizeFloat64,
			ReduceContainsFloat64(BroadcastFloat64(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsFloat64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		for _, target := range []float64{1.5, 3.5, 100.5, 9999.5} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[float64, PartialFloat64s, bool](
					ro.FromSlice(input),
					VectorizeFloat64,
					ReduceContainsFloat64(BroadcastFloat64(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %v", size, target)
		}
	}
}

func TestAddWithFloat64ZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeFloat64[PartialFloat64s](ro.FromSlice(rampFloat64(20)))
	right := VectorizeFloat64[PartialFloat64s](ro.FromSlice(rampFloat64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialFloat64s, []float64, float64](
			AddWithFloat64(right)(left),
			ro.Map(func(v PartialFloat64s) []float64 { return v.Values() }),
			ro.Flatten[float64](),
		),
	)
	assert.NoError(t, err)

	want := make([]float64, 20)
	for i, v := range rampFloat64(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapFloat64(t *testing.T) {
	t.Parallel()

	input := rampFloat64(10)

	values, err := ro.Collect(
		ro.Pipe3[float64, PartialFloat64s, []float64, float64](
			ro.FromSlice(input),
			VectorizeFloat64,
			ro.Map(func(v PartialFloat64s) []float64 {
				return v.Add(BroadcastFloat64(42)).Min(BroadcastFloat64(50)).Values()
			}),
			ro.Flatten[float64](),
		),
	)
	assert.NoError(t, err)

	want := make([]float64, len(input))
	for i, v := range input {
		want[i] = min(v+42, 50)
	}

	assert.Equal(t, want, values)
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsFloat64(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()
	batch := rampFloat64(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadFloat64s(batch)), AddFloat64(simd.BroadcastFloat64s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]float64
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

func TestEmptyAndErrorSourcesFloat64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[float64, PartialFloat64s, float64](ro.Empty[float64](), VectorizeFloat64, ReduceSumFloat64),
	)
	assert.NoError(t, err)
	assert.Equal(t, []float64{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeFloat64[PartialFloat64s](ro.Empty[float64]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialFloat64s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeFloat64[PartialFloat64s](ro.Throw[float64](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[float64, PartialFloat64s, float64](ro.Throw[float64](assert.AnError), VectorizeFloat64, ReduceSumFloat64),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}
