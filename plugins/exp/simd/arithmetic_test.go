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
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// Differential tests for Add, Sub, Mul and Div against the equivalent core ro
// pipeline, across a sweep of input sizes that surrounds every lane boundary.

func TestAddInt8WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt8[PartialInt8s]()(ro.FromSlice(rampInt8(20)))
	right := VectorizeInt8[PartialInt8s]()(ro.FromSlice(rampInt8(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt8s, []int8, int8](
			AddInt8With(right)(left),
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

func TestAddInt16WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt16[PartialInt16s]()(ro.FromSlice(rampInt16(20)))
	right := VectorizeInt16[PartialInt16s]()(ro.FromSlice(rampInt16(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt16s, []int16, int16](
			AddInt16With(right)(left),
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

func TestAddInt32WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt32[PartialInt32s]()(ro.FromSlice(rampInt32(20)))
	right := VectorizeInt32[PartialInt32s]()(ro.FromSlice(rampInt32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt32s, []int32, int32](
			AddInt32With(right)(left),
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

func TestAddInt64WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeInt64[PartialInt64s]()(ro.FromSlice(rampInt64(20)))
	right := VectorizeInt64[PartialInt64s]()(ro.FromSlice(rampInt64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialInt64s, []int64, int64](
			AddInt64With(right)(left),
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

func TestAddUint8WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint8[PartialUint8s]()(ro.FromSlice(rampUint8(20)))
	right := VectorizeUint8[PartialUint8s]()(ro.FromSlice(rampUint8(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint8s, []uint8, uint8](
			AddUint8With(right)(left),
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

func TestAddUint16WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint16[PartialUint16s]()(ro.FromSlice(rampUint16(20)))
	right := VectorizeUint16[PartialUint16s]()(ro.FromSlice(rampUint16(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint16s, []uint16, uint16](
			AddUint16With(right)(left),
			ro.Map(func(v PartialUint16s) []uint16 { return v.Values() }),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	want := make([]uint16, 20)
	for i, v := range rampUint16(20) {
		want[i] = v * 2
	}

	assert.Equal(t, want, values)
}

func TestAddUint32WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint32[PartialUint32s]()(ro.FromSlice(rampUint32(20)))
	right := VectorizeUint32[PartialUint32s]()(ro.FromSlice(rampUint32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint32s, []uint32, uint32](
			AddUint32With(right)(left),
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

func TestAddUint64WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeUint64[PartialUint64s]()(ro.FromSlice(rampUint64(20)))
	right := VectorizeUint64[PartialUint64s]()(ro.FromSlice(rampUint64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialUint64s, []uint64, uint64](
			AddUint64With(right)(left),
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

func TestAddFloat32WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeFloat32[PartialFloat32s]()(ro.FromSlice(rampFloat32(20)))
	right := VectorizeFloat32[PartialFloat32s]()(ro.FromSlice(rampFloat32(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialFloat32s, []float32, float32](
			AddFloat32With(right)(left),
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

func TestAddFloat64WithZipsTwoStreams(t *testing.T) {
	t.Parallel()

	left := VectorizeFloat64[PartialFloat64s]()(ro.FromSlice(rampFloat64(20)))
	right := VectorizeFloat64[PartialFloat64s]()(ro.FromSlice(rampFloat64(20)))

	values, err := ro.Collect(
		ro.Pipe2[PartialFloat64s, []float64, float64](
			AddFloat64With(right)(left),
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

// Padded lanes are zero-filled, so an unmasked Div of two short vectors would compute
// 0/0 and leave NaN behind — which a later Min or Max would then propagate into a
// valid lane.
func TestDivFloat32WithLeavesNoNaNInPadding(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat32()

	left := VectorizeFloat32[PartialFloat32s]()(ro.FromSlice([]float32{1}))
	right := VectorizeFloat32[PartialFloat32s]()(ro.FromSlice([]float32{2}))

	vectors, err := ro.Collect(DivFloat32With(right)(left))
	assert.NoError(t, err)
	assert.Len(t, vectors, 1)

	assert.Equal(t, []float32{0.5}, vectors[0].Values())

	for lane := 1; lane < lanes; lane++ {
		assert.False(t, isNaNFloat32(vectors[0].rawLane(lane)),
			"padded lane %d must not hold the NaN that 0/0 produces", lane)
		assert.Zero(t, vectors[0].rawLane(lane), "padded lane %d", lane)
	}
}

// Padded lanes are zero-filled, so an unmasked Div of two short vectors would compute
// 0/0 and leave NaN behind — which a later Min or Max would then propagate into a
// valid lane.
func TestDivFloat64WithLeavesNoNaNInPadding(t *testing.T) {
	t.Parallel()

	lanes := lanesFloat64()

	left := VectorizeFloat64[PartialFloat64s]()(ro.FromSlice([]float64{1}))
	right := VectorizeFloat64[PartialFloat64s]()(ro.FromSlice([]float64{2}))

	vectors, err := ro.Collect(DivFloat64With(right)(left))
	assert.NoError(t, err)
	assert.Len(t, vectors, 1)

	assert.Equal(t, []float64{0.5}, vectors[0].Values())

	for lane := 1; lane < lanes; lane++ {
		assert.False(t, isNaNFloat64(vectors[0].rawLane(lane)),
			"padded lane %d must not hold the NaN that 0/0 produces", lane)
		assert.Zero(t, vectors[0].rawLane(lane), "padded lane %d", lane)
	}
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

func TestAddUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint16) uint16 { return v + 42 })),
		)
		assert.NoError(t, err)

		got := collectUint16Values[PartialUint16s](t, input, AddUint16(BroadcastUint16(42)))

		assert.Equal(t, want, got, "size %d", size)
	}
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

func TestSubUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(
			ro.Pipe1(ro.FromSlice(input), ro.Map(func(v uint16) uint16 { return v - 7 })),
		)
		assert.NoError(t, err)

		got := collectUint16Values[PartialUint16s](t, input, SubUint16(BroadcastUint16(7)))

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

// Dividing by zero is not an error in Go and must not become one here.
func TestDivFloat32ByZeroFollowsGo(t *testing.T) {
	t.Parallel()

	zero := float32(0)
	want := []float32{1 / zero, -1 / zero, zero / zero}

	got := collectFloat32Values[PartialFloat32s](t, []float32{1, -1, 0}, DivFloat32(BroadcastFloat32(0)))

	assertLanesEqualFloat32(t, want, got, "division by zero")
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
func TestDivFloat64ByZeroFollowsGo(t *testing.T) {
	t.Parallel()

	zero := float64(0)
	want := []float64{1 / zero, -1 / zero, zero / zero}

	got := collectFloat64Values[PartialFloat64s](t, []float64{1, -1, 0}, DivFloat64(BroadcastFloat64(0)))

	assertLanesEqualFloat64(t, want, got, "division by zero")
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

// Add is covered over stdlib vectors in partial_test.go; the rest of the arithmetic
// family needs the same guarantee, since each one's constraint is satisfied separately.
func TestArithmeticOperatorsAcceptStdlibVectors(t *testing.T) {
	t.Parallel()

	batch := rampInt8(lanesInt8())

	subtracted, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt8s(batch)), SubInt8(simd.BroadcastInt8s(1))),
	)
	assert.NoError(t, err)

	multiplied, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt8s(batch)), MulInt8(simd.BroadcastInt8s(2))),
	)
	assert.NoError(t, err)

	var subOut, mulOut [maxLanes]int8
	n := subtracted[0].StorePart(subOut[:])
	multiplied[0].StorePart(mulOut[:])

	for i := range n {
		assert.Equal(t, batch[i]-1, subOut[i], "Sub lane %d", i)
		assert.Equal(t, batch[i]*2, mulOut[i], "Mul lane %d", i)
	}

	// Div has no integer form, so it is checked on the float type instead.
	floats := rampFloat64(lanesFloat64())

	divided, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadFloat64s(floats)), DivFloat64(simd.BroadcastFloat64s(2))),
	)
	assert.NoError(t, err)

	var divOut [maxLanes]float64
	divided[0].StorePart(divOut[:])

	for i := range floats {
		assert.Equal(t, floats[i]/2, divOut[i], "Div lane %d", i)
	}
}
