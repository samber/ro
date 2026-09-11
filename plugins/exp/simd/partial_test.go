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

// Tests for the Partial types themselves rather than any one operator: that padded
// lanes never leak into the output or get modified, that methods chain through ro.Map,
// and that the standard library's own vector types satisfy the same constraints.

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

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedInt16(t *testing.T) {
	t.Parallel()

	input := rampInt16(3)

	got := collectInt16Values[PartialInt16s](t, input, AddInt16(BroadcastInt16(100)))

	assert.Equal(t, []int16{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsInt16(t *testing.T) {
	t.Parallel()

	lanes := lanesInt16()

	tail := PartialInt16s{}.LoadPart([]int16{1, 2, 3}, 3).Add(BroadcastInt16(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, int16(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedInt32(t *testing.T) {
	t.Parallel()

	input := rampInt32(3)

	got := collectInt32Values[PartialInt32s](t, input, AddInt32(BroadcastInt32(100)))

	assert.Equal(t, []int32{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsInt32(t *testing.T) {
	t.Parallel()

	lanes := lanesInt32()

	tail := PartialInt32s{}.LoadPart([]int32{1, 2, 3}, 3).Add(BroadcastInt32(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, int32(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
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

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint8(t *testing.T) {
	t.Parallel()

	input := rampUint8(3)

	got := collectUint8Values[PartialUint8s](t, input, AddUint8(BroadcastUint8(100)))

	assert.Equal(t, []uint8{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()

	tail := PartialUint8s{}.LoadPart([]uint8{1, 2, 3}, 3).Add(BroadcastUint8(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint8(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint16(t *testing.T) {
	t.Parallel()

	input := rampUint16(3)

	got := collectUint16Values[PartialUint16s](t, input, AddUint16(BroadcastUint16(100)))

	assert.Equal(t, []uint16{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()

	tail := PartialUint16s{}.LoadPart([]uint16{1, 2, 3}, 3).Add(BroadcastUint16(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint16(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
	}
}

// Padded lanes must not leak into the output. Adding to a short tail is where a
// missing mask would show up as extra values.
func TestPaddedLanesNeverEmittedUint32(t *testing.T) {
	t.Parallel()

	input := rampUint32(3)

	got := collectUint32Values[PartialUint32s](t, input, AddUint32(BroadcastUint32(100)))

	assert.Equal(t, []uint32{101, 102, 103}, got)
	assert.Len(t, got, 3, "a 3-item input must never emit a full vector's worth of lanes")
}

// The mask's whole job is keeping padded lanes at their pre-operation value. That is
// invisible through Values or StorePart, which stop at the valid count, so assert it
// against the raw vector — otherwise dropping the mask entirely would pass the suite.
func TestPaddedLanesStayUntouchedByOperationsUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()

	tail := PartialUint32s{}.LoadPart([]uint32{1, 2, 3}, 3).Add(BroadcastUint32(100))

	assert.Equal(t, 3, tail.Count())

	for lane := 3; lane < lanes; lane++ {
		assert.Zero(t, tail.rawLane(lane),
			"padded lane %d must keep the zero LoadPart left, not 0+100", lane)
	}

	for lane := range 3 {
		assert.Equal(t, uint32(lane+1+100), tail.rawLane(lane), "valid lane %d", lane)
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

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMap(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int8, PartialInt8s, []int8, int8](
			ro.FromSlice(rampInt8(10)),
			VectorizeInt8[PartialInt8s](),
			ro.Map(func(v PartialInt8s) []int8 {
				return v.Add(BroadcastInt8(42)).Min(BroadcastInt8(50)).Values()
			}),
			ro.Flatten[int8](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int8{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int16, PartialInt16s, []int16, int16](
			ro.FromSlice(rampInt16(10)),
			VectorizeInt16[PartialInt16s](),
			ro.Map(func(v PartialInt16s) []int16 {
				return v.Add(BroadcastInt16(42)).Min(BroadcastInt16(50)).Values()
			}),
			ro.Flatten[int16](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int16{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int32, PartialInt32s, []int32, int32](
			ro.FromSlice(rampInt32(10)),
			VectorizeInt32[PartialInt32s](),
			ro.Map(func(v PartialInt32s) []int32 {
				return v.Add(BroadcastInt32(42)).Min(BroadcastInt32(50)).Values()
			}),
			ro.Flatten[int32](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int32{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapInt64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[int64, PartialInt64s, []int64, int64](
			ro.FromSlice(rampInt64(10)),
			VectorizeInt64[PartialInt64s](),
			ro.Map(func(v PartialInt64s) []int64 {
				return v.Add(BroadcastInt64(42)).Min(BroadcastInt64(50)).Values()
			}),
			ro.Flatten[int64](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []int64{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint8(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint8, PartialUint8s, []uint8, uint8](
			ro.FromSlice(rampUint8(10)),
			VectorizeUint8[PartialUint8s](),
			ro.Map(func(v PartialUint8s) []uint8 {
				return v.Add(BroadcastUint8(42)).Min(BroadcastUint8(50)).Values()
			}),
			ro.Flatten[uint8](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint8{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint16(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint16, PartialUint16s, []uint16, uint16](
			ro.FromSlice(rampUint16(10)),
			VectorizeUint16[PartialUint16s](),
			ro.Map(func(v PartialUint16s) []uint16 {
				return v.Add(BroadcastUint16(42)).Min(BroadcastUint16(50)).Values()
			}),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint16{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint32(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint32, PartialUint32s, []uint32, uint32](
			ro.FromSlice(rampUint32(10)),
			VectorizeUint32[PartialUint32s](),
			ro.Map(func(v PartialUint32s) []uint32 {
				return v.Add(BroadcastUint32(42)).Min(BroadcastUint32(50)).Values()
			}),
			ro.Flatten[uint32](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint32{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapUint64(t *testing.T) {
	t.Parallel()

	values, err := ro.Collect(
		ro.Pipe3[uint64, PartialUint64s, []uint64, uint64](
			ro.FromSlice(rampUint64(10)),
			VectorizeUint64[PartialUint64s](),
			ro.Map(func(v PartialUint64s) []uint64 {
				return v.Add(BroadcastUint64(42)).Min(BroadcastUint64(50)).Values()
			}),
			ro.Flatten[uint64](),
		),
	)
	assert.NoError(t, err)

	assert.Equal(t, []uint64{43, 44, 45, 46, 47, 48, 49, 50, 50, 50}, values)
}

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapFloat32(t *testing.T) {
	t.Parallel()

	input := rampFloat32(10)

	values, err := ro.Collect(
		ro.Pipe3[float32, PartialFloat32s, []float32, float32](
			ro.FromSlice(input),
			VectorizeFloat32[PartialFloat32s](),
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

// Method chaining through core ro.Map is the alternative to stacking operators, and
// needs no type arguments at all.
func TestMethodChainingViaMapFloat64(t *testing.T) {
	t.Parallel()

	input := rampFloat64(10)

	values, err := ro.Collect(
		ro.Pipe3[float64, PartialFloat64s, []float64, float64](
			ro.FromSlice(input),
			VectorizeFloat64[PartialFloat64s](),
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

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsInt16(t *testing.T) {
	t.Parallel()

	lanes := lanesInt16()
	batch := rampInt16(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt16s(batch)), AddInt16(simd.BroadcastInt16s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]int16
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsInt32(t *testing.T) {
	t.Parallel()

	lanes := lanesInt32()
	batch := rampInt32(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadInt32s(batch)), AddInt32(simd.BroadcastInt32s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]int32
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint8(t *testing.T) {
	t.Parallel()

	lanes := lanesUint8()
	batch := rampUint8(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint8s(batch)), AddUint8(simd.BroadcastUint8s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint8
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint16(t *testing.T) {
	t.Parallel()

	lanes := lanesUint16()
	batch := rampUint16(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint16s(batch)), AddUint16(simd.BroadcastUint16s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint16
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
}

// The same operator must accept the standard library's own vector type.
func TestOperatorsAcceptStdlibVectorsUint32(t *testing.T) {
	t.Parallel()

	lanes := lanesUint32()
	batch := rampUint32(lanes)

	values, err := ro.Collect(
		ro.Pipe1(ro.Just(simd.LoadUint32s(batch)), AddUint32(simd.BroadcastUint32s(100))),
	)
	assert.NoError(t, err)

	var out [maxLanes]uint32
	n := values[0].StorePart(out[:])

	for i := range n {
		assert.Equal(t, batch[i]+100, out[i], "lane %d", i)
	}
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
