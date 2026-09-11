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

// Differential tests for ReduceSum, ReduceMin and ReduceMax against core ro.Sum,
// ro.Min and ro.Max.

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

func TestReduceSumUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceSumUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceSumUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Sum[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceSumUint32),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
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

func TestReduceMinUint16MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMinUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMinUint32MatchesCore(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Min[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMinUint32),
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

func TestReduceMaxUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint16(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint16]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, ReduceMaxUint16),
		)
		assert.NoError(t, err)

		assert.Equal(t, want, got, "size %d", size)
	}
}

func TestReduceMaxUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		if size == 0 {
			continue // ro.Max emits an unguarded zero on empty; rosimd emits nothing.
		}

		input := rampUint32(size)

		want, err := ro.Collect(ro.Pipe1(ro.FromSlice(input), ro.Max[uint32]()))
		assert.NoError(t, err)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, ReduceMaxUint32),
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

// The reductions are generic over the same constraint as the element-wise operators, so
// they too must accept the standard library's vector type, not only the Partial types.
//
// A full stdlib vector has no validity mask and no padding, which is exactly why this is
// worth pinning: every lane counts, including any that a Partial would have masked off.
func TestReductionsAcceptStdlibVectors(t *testing.T) {
	t.Parallel()

	batch := make([]int8, lanesInt8())
	for i := range batch {
		batch[i] = 1
	}
	batch[0] = 5
	batch[len(batch)-1] = -3

	var wantSum int8
	for _, value := range batch {
		wantSum += value
	}

	sum, err := ro.Collect(ReduceSumInt8[simd.Int8s](ro.Just(simd.LoadInt8s(batch))))
	assert.NoError(t, err)
	assert.Equal(t, []int8{wantSum}, sum)

	smallest, err := ro.Collect(ReduceMinInt8[simd.Int8s](ro.Just(simd.LoadInt8s(batch))))
	assert.NoError(t, err)
	assert.Equal(t, []int8{-3}, smallest)

	largest, err := ro.Collect(ReduceMaxInt8[simd.Int8s](ro.Just(simd.LoadInt8s(batch))))
	assert.NoError(t, err)
	assert.Equal(t, []int8{5}, largest)
}
