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
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// Tests for Min, Max and Clamp.
//
// Two behaviours get special attention: NaN, which the element-wise float operators
// force into the result because hardware disagrees about it, and unsigned ordering,
// where a value above the signed maximum must still compare as large.

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
