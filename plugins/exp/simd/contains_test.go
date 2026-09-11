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

// Tests for the element-wise Contains method, which returns a per-lane mask, and for
// the ReduceContains operator, which collapses a stream to one bool.
//
// Searching for zero is the case the validity mask exists for: padding is zero-filled,
// so an unmasked compare would report a match on any short tail.

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

func TestReduceContainsInt16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepInt16() {
		input := rampInt16(size)

		for _, target := range []int16{1, 3, 100, 32767} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[int16, PartialInt16s, bool](
					ro.FromSlice(input),
					VectorizeInt16,
					ReduceContainsInt16(BroadcastInt16(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsInt16ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[int16, PartialInt16s, bool](
			ro.FromSlice([]int16{1, 2, 3}),
			VectorizeInt16,
			ReduceContainsInt16(BroadcastInt16(0)),
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

func TestReduceContainsUint8(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint8() {
		input := rampUint8(size)

		for _, target := range []uint8{1, 3, 100, 255} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint8, PartialUint8s, bool](
					ro.FromSlice(input),
					VectorizeUint8,
					ReduceContainsUint8(BroadcastUint8(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint8ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint8, PartialUint8s, bool](
			ro.FromSlice([]uint8{1, 2, 3}),
			VectorizeUint8,
			ReduceContainsUint8(BroadcastUint8(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint16(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint16() {
		input := rampUint16(size)

		for _, target := range []uint16{1, 3, 100, 65535} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint16, PartialUint16s, bool](
					ro.FromSlice(input),
					VectorizeUint16,
					ReduceContainsUint16(BroadcastUint16(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint16ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint16, PartialUint16s, bool](
			ro.FromSlice([]uint16{1, 2, 3}),
			VectorizeUint16,
			ReduceContainsUint16(BroadcastUint16(0)),
		),
	)
	assert.NoError(t, err)
	assert.Equal(t, []bool{false}, got, "padded lanes must not match a search for zero")
}

func TestReduceContainsUint32(t *testing.T) {
	t.Parallel()

	for _, size := range sizeSweepUint32() {
		input := rampUint32(size)

		for _, target := range []uint32{1, 3, 100, 4294967295} {
			want := slices.Contains(input, target)

			got, err := ro.Collect(
				ro.Pipe2[uint32, PartialUint32s, bool](
					ro.FromSlice(input),
					VectorizeUint32,
					ReduceContainsUint32(BroadcastUint32(target)),
				),
			)
			assert.NoError(t, err)
			assert.Equal(t, []bool{want}, got, "size %d target %d", size, target)
		}
	}
}

// Searching for zero is the case the mask exists for: LoadPart zero-fills padded
// lanes, so an unmasked compare would report a match on any short tail.
func TestReduceContainsUint32ZeroIsNotFoundInPadding(t *testing.T) {
	t.Parallel()

	got, err := ro.Collect(
		ro.Pipe2[uint32, PartialUint32s, bool](
			ro.FromSlice([]uint32{1, 2, 3}),
			VectorizeUint32,
			ReduceContainsUint32(BroadcastUint32(0)),
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
func TestContainsInt16MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialInt16s{}.LoadPart([]int16{0, 1, 0}, 3)

	lanes := maskLanesInt16(vector.Contains(BroadcastInt16(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsInt16ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialInt16s{}.LoadPart([]int16{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastInt16(7)), BroadcastInt16(0))

	assert.Equal(t, []int16{7, 0, 7, 0}, picked.Values(),
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
func TestContainsUint8MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint8s{}.LoadPart([]uint8{0, 1, 0}, 3)

	lanes := maskLanesUint8(vector.Contains(BroadcastUint8(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint8ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint8s{}.LoadPart([]uint8{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint8(7)), BroadcastUint8(0))

	assert.Equal(t, []uint8{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsUint16MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint16s{}.LoadPart([]uint16{0, 1, 0}, 3)

	lanes := maskLanesUint16(vector.Contains(BroadcastUint16(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint16ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint16s{}.LoadPart([]uint16{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint16(7)), BroadcastUint16(0))

	assert.Equal(t, []uint16{7, 0, 7, 0}, picked.Values(),
		"lanes equal to 7 keep their value, the rest take the replacement")
}

// The mask is intersected with validity, so padding is never reported as a match —
// even when searching for the very zero that padding is filled with.
func TestContainsUint32MaskExcludesPadding(t *testing.T) {
	t.Parallel()

	vector := PartialUint32s{}.LoadPart([]uint32{0, 1, 0}, 3)

	lanes := maskLanesUint32(vector.Contains(BroadcastUint32(0)))

	assert.Equal(t, []bool{true, false, true}, lanes[:3], "valid lanes equal to zero match")

	for lane := 3; lane < len(lanes); lane++ {
		assert.False(t, lanes[lane],
			"padded lane %d is zero-filled but must not be reported as a match", lane)
	}
}

// Contains is element-wise: it reports which lanes match, not whether any did.
func TestContainsUint32ReturnsPerLaneMask(t *testing.T) {
	t.Parallel()

	vector := PartialUint32s{}.LoadPart([]uint32{7, 1, 7, 2}, 4)

	// Select proves the mask is usable, which is the point of returning one.
	picked := vector.Select(vector.Contains(BroadcastUint32(7)), BroadcastUint32(0))

	assert.Equal(t, []uint32{7, 0, 7, 0}, picked.Values(),
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

// On a full vector every lane is valid, so intersecting with the validity mask must be a
// no-op and Contains must agree lane for lane with the stdlib compare it is built on.
// This pins the mask down against an independent construction rather than against
// Select, which re-applies the validity mask and would hide a mistake in it.
func TestContainsMatchesAStdlibCompareOnAFullVector(t *testing.T) {
	t.Parallel()

	batch := make([]int8, lanesInt8())
	for i := range batch {
		batch[i] = int8(i % 3)
	}

	want := maskLanesInt8(simd.LoadInt8s(batch).Equal(simd.BroadcastInt8s(2)))
	got := maskLanesInt8(PartialInt8s{}.Load(batch).Contains(BroadcastInt8(2)))

	assert.Equal(t, want, got)
}
