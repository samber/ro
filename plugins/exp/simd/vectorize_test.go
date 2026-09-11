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

// Tests for batching a scalar stream into vectors: that values round-trip unchanged,
// that batches have the shape the lane width implies, and that empty and failing
// sources behave.

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

func TestVectorizeRoundTrip(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		got := collectInt8Values[PartialInt8s](t, input)

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

func TestVectorizeRoundTripInt16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		got := collectInt16Values[PartialInt16s](t, input)

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

func TestVectorizeRoundTripInt32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		got := collectInt32Values[PartialInt32s](t, input)

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

func TestVectorizeRoundTripInt64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		got := collectInt64Values[PartialInt64s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()

	// 35 items at 16 lanes is two full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint8[PartialUint8s](ro.FromSlice(rampUint8(35))))
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

func TestVectorizeRoundTripUint8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		got := collectUint8Values[PartialUint8s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()

	// 35 items at 8 lanes is four full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint16[PartialUint16s](ro.FromSlice(rampUint16(35))))
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

func TestVectorizeRoundTripUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		got := collectUint16Values[PartialUint16s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
	}
}

func TestVectorizeBatchShapesUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()

	// 35 items at 4 lanes is eight full vectors plus a 3-lane tail.
	vectors, err := ro.Collect(VectorizeUint32[PartialUint32s](ro.FromSlice(rampUint32(35))))
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

func TestVectorizeRoundTripUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		got := collectUint32Values[PartialUint32s](t, input)

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

func TestVectorizeRoundTripUint64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		got := collectUint64Values[PartialUint64s](t, input)

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

func TestVectorizeRoundTripFloat32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		got := collectFloat32Values[PartialFloat32s](t, input)

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

func TestVectorizeRoundTripFloat64(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		got := collectFloat64Values[PartialFloat64s](t, input)

		assert.Equal(t, input, got, "size %d must round-trip unchanged", size)
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

func TestEmptyAndErrorSourcesUint8(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.Empty[uint8](), VectorizeUint8, ReduceSumUint8),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint8{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint8[PartialUint8s](ro.Empty[uint8]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint8s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint8[PartialUint8s](ro.Throw[uint8](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, uint8](ro.Throw[uint8](assert.AnError), VectorizeUint8, ReduceSumUint8),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

func TestEmptyAndErrorSourcesUint16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.Empty[uint16](), VectorizeUint16, ReduceSumUint16),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint16{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint16[PartialUint16s](ro.Empty[uint16]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint16s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint16[PartialUint16s](ro.Throw[uint16](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, uint16](ro.Throw[uint16](assert.AnError), VectorizeUint16, ReduceSumUint16),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

func TestEmptyAndErrorSourcesUint32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.Empty[uint32](), VectorizeUint32, ReduceSumUint32),
	)
	assert.NoError(t, err)
	assert.Equal(t, []uint32{0}, values, "an empty stream sums to zero, as ro.Sum does")

	vectors, err := ro.Collect(VectorizeUint32[PartialUint32s](ro.Empty[uint32]()))
	assert.NoError(t, err)
	assert.Equal(t, []PartialUint32s{}, vectors, "an empty stream must not emit a padding-only vector")

	_, err = ro.Collect(VectorizeUint32[PartialUint32s](ro.Throw[uint32](assert.AnError)))
	assert.EqualError(t, err, assert.AnError.Error())

	_, err = ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, uint32](ro.Throw[uint32](assert.AnError), VectorizeUint32, ReduceSumUint32),
	)
	assert.EqualError(t, err, assert.AnError.Error())
}

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

// Vectorize sizes its batches from the zero value's Len, so that must report the width
// the architecture really has. A cached or placeholder width would silently batch at the
// wrong size — the failure mode the package doc warns about for simd-dependent globals.
func TestPartialLenMatchesStdlibWidth(t *testing.T) {
	t.Parallel()

	assert.Equal(t, simd.BroadcastInt8s(0).Len(), PartialInt8s{}.Len())
	assert.Equal(t, simd.BroadcastInt32s(0).Len(), PartialInt32s{}.Len())
	assert.Equal(t, simd.BroadcastUint64s(0).Len(), PartialUint64s{}.Len())
	assert.Equal(t, simd.BroadcastFloat64s(0).Len(), PartialFloat64s{}.Len())
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenInt8RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		got, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, int8](ro.FromSlice(input), VectorizeInt8, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarInt8MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt8() {
		input := rampInt8(size)

		batches, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, []int8](ro.FromSlice(input), VectorizeInt8, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[int8, PartialInt8s, int](
				ro.FromSlice(input),
				VectorizeInt8,
				ro.Map(func(v PartialInt8s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []int8{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenInt16RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		got, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, int16](ro.FromSlice(input), VectorizeInt16, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarInt16MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		batches, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, []int16](ro.FromSlice(input), VectorizeInt16, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[int16, PartialInt16s, int](
				ro.FromSlice(input),
				VectorizeInt16,
				ro.Map(func(v PartialInt16s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []int16{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenInt32RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		got, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, int32](ro.FromSlice(input), VectorizeInt32, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarInt32MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt32() {
		input := rampInt32(size)

		batches, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, []int32](ro.FromSlice(input), VectorizeInt32, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[int32, PartialInt32s, int](
				ro.FromSlice(input),
				VectorizeInt32,
				ro.Map(func(v PartialInt32s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []int32{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenInt64RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		got, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, int64](ro.FromSlice(input), VectorizeInt64, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarInt64MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt64() {
		input := rampInt64(size)

		batches, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, []int64](ro.FromSlice(input), VectorizeInt64, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[int64, PartialInt64s, int](
				ro.FromSlice(input),
				VectorizeInt64,
				ro.Map(func(v PartialInt64s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []int64{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenUint8RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		got, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, uint8](ro.FromSlice(input), VectorizeUint8, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarUint8MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		batches, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, []uint8](ro.FromSlice(input), VectorizeUint8, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[uint8, PartialUint8s, int](
				ro.FromSlice(input),
				VectorizeUint8,
				ro.Map(func(v PartialUint8s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []uint8{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenUint16RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		got, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, uint16](ro.FromSlice(input), VectorizeUint16, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarUint16MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		batches, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, []uint16](ro.FromSlice(input), VectorizeUint16, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[uint16, PartialUint16s, int](
				ro.FromSlice(input),
				VectorizeUint16,
				ro.Map(func(v PartialUint16s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []uint16{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenUint32RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		got, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, uint32](ro.FromSlice(input), VectorizeUint32, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarUint32MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		batches, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, []uint32](ro.FromSlice(input), VectorizeUint32, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[uint32, PartialUint32s, int](
				ro.FromSlice(input),
				VectorizeUint32,
				ro.Map(func(v PartialUint32s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []uint32{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenUint64RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		got, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, uint64](ro.FromSlice(input), VectorizeUint64, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarUint64MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint64() {
		input := rampUint64(size)

		batches, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, []uint64](ro.FromSlice(input), VectorizeUint64, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[uint64, PartialUint64s, int](
				ro.FromSlice(input),
				VectorizeUint64,
				ro.Map(func(v PartialUint64s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []uint64{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenFloat32RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		got, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, float32](ro.FromSlice(input), VectorizeFloat32, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarFloat32MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat32() {
		input := rampFloat32(size)

		batches, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, []float32](ro.FromSlice(input), VectorizeFloat32, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[float32, PartialFloat32s, int](
				ro.FromSlice(input),
				VectorizeFloat32,
				ro.Map(func(v PartialFloat32s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []float32{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// Flatten is Vectorize's inverse, so a stream that goes in and comes back out must be
// unchanged at every size — the short final batch above all, where a forgotten mask
// would emit padding as if it were data.
func TestFlattenFloat64RoundTrips(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		got, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, float64](ro.FromSlice(input), VectorizeFloat64, Flatten),
		)
		assert.NoError(t, err)

		assert.Equal(t, input, got, "size %d must survive the round trip", size)
	}
}

// ToScalar emits one slice per vector, each exactly as long as that vector's valid lane
// count, and the slices concatenated are the original stream.
func TestToScalarFloat64MatchesBatches(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepFloat64() {
		input := rampFloat64(size)

		batches, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, []float64](ro.FromSlice(input), VectorizeFloat64, ToScalar),
		)
		assert.NoError(t, err)

		counts, err := ro.Collect(
			ro.Pipe2[float64, PartialFloat64s, int](
				ro.FromSlice(input),
				VectorizeFloat64,
				ro.Map(func(v PartialFloat64s) int { return v.Count() }),
			),
		)
		assert.NoError(t, err)

		joined := []float64{}
		for i, batch := range batches {
			assert.Len(t, batch, counts[i], "size %d batch %d", size, i)
			joined = append(joined, batch...)
		}

		assert.Equal(t, input, joined, "size %d", size)
	}
}

// ToScalar and Flatten ask only that a vector can report its lanes, not that it can do
// arithmetic, so they accept more types than the operators do — simd.Int64s included,
// which AddInt64 and the rest reject for want of Min and Max.
func TestExitOperatorsAcceptStdlibVectors(t *testing.T) {
	t.Parallel()

	batch := rampInt8(lanesInt8())

	flattened, err := ro.Collect(Flatten(ro.Just(simd.LoadInt8s(batch))))
	assert.NoError(t, err)
	assert.Equal(t, batch, flattened)

	sliced, err := ro.Collect(ToScalar(ro.Just(simd.LoadInt8s(batch))))
	assert.NoError(t, err)
	assert.Equal(t, [][]int8{batch}, sliced)

	wide := rampInt64(lanesInt64())

	wideFlattened, err := ro.Collect(Flatten(ro.Just(simd.LoadInt64s(wide))))
	assert.NoError(t, err)
	assert.Equal(t, wide, wideFlattened,
		"simd.Int64s cannot satisfy Int64Vector, but it can report its lanes")
}
