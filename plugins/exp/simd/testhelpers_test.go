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

// Shared fixtures for the tests in this package, one set per element type.
//
// collectXxxValues must be generic over V rather than naming a Partial type in its
// signature: a non-generic function holding a concrete simd-containing type in a
// parameter fails to compile, because the specializer clones the type and the
// synthesized dispatcher cannot convert the argument to the clone's type.

func lanesInt8() int { return simd.BroadcastInt8s(0).Len() }

// sizeSweepInt8 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt8() []int {
	lanes := lanesInt8()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt8 builds 1, 2, 3... wrapping within int8 so long inputs stay in range.
func rampInt8(size int) []int8 {
	out := make([]int8, size)
	for i := range out {
		out[i] = int8(i%100 + 1)
	}

	return out
}

// collectInt8Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt8s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt8Values[V Int8Buffer[V]](t *testing.T, input []int8, operators ...func(ro.Observable[V]) ro.Observable[V]) []int8 {
	t.Helper()

	vectors := VectorizeInt8[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int8, int8](
			vectors,
			ro.Map(func(v V) []int8 {
				var buffer [maxLanes]int8
				n := v.StorePart(buffer[:])

				return append([]int8{}, buffer[:n]...)
			}),
			ro.Flatten[int8](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt8 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt8(mask simd.Mask8s) []bool {
	lanes := lanesInt8()

	var buf [maxLanes]int8
	mask.ToInt8s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func lanesInt16() int { return simd.BroadcastInt16s(0).Len() }

// sizeSweepInt16 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt16() []int {
	lanes := lanesInt16()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt16 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampInt16(size int) []int16 {
	out := make([]int16, size)
	for i := range out {
		out[i] = int16(i%1000 + 1)
	}

	return out
}

// collectInt16Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt16s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt16Values[V Int16Buffer[V]](t *testing.T, input []int16, operators ...func(ro.Observable[V]) ro.Observable[V]) []int16 {
	t.Helper()

	vectors := VectorizeInt16[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int16, int16](
			vectors,
			ro.Map(func(v V) []int16 {
				var buffer [maxLanes]int16
				n := v.StorePart(buffer[:])

				return append([]int16{}, buffer[:n]...)
			}),
			ro.Flatten[int16](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt16 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt16(mask simd.Mask16s) []bool {
	lanes := lanesInt16()

	var buf [maxLanes]int16
	mask.ToInt16s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func lanesInt32() int { return simd.BroadcastInt32s(0).Len() }

// sizeSweepInt32 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepInt32() []int {
	lanes := lanesInt32()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampInt32 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampInt32(size int) []int32 {
	out := make([]int32, size)
	for i := range out {
		out[i] = int32(i%1000 + 1)
	}

	return out
}

// collectInt32Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialInt32s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectInt32Values[V Int32Buffer[V]](t *testing.T, input []int32, operators ...func(ro.Observable[V]) ro.Observable[V]) []int32 {
	t.Helper()

	vectors := VectorizeInt32[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []int32, int32](
			vectors,
			ro.Map(func(v V) []int32 {
				var buffer [maxLanes]int32
				n := v.StorePart(buffer[:])

				return append([]int32{}, buffer[:n]...)
			}),
			ro.Flatten[int32](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesInt32 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesInt32(mask simd.Mask32s) []bool {
	lanes := lanesInt32()

	var buf [maxLanes]int32
	mask.ToInt32s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

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

	vectors := VectorizeInt64[V]()(ro.FromSlice(input))
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

func lanesUint8() int { return simd.BroadcastUint8s(0).Len() }

// sizeSweepUint8 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint8() []int {
	lanes := lanesUint8()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint8 builds 1, 2, 3... wrapping within uint8 so long inputs stay in range.
func rampUint8(size int) []uint8 {
	out := make([]uint8, size)
	for i := range out {
		out[i] = uint8(i%100 + 1)
	}

	return out
}

// collectUint8Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint8s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint8Values[V Uint8Buffer[V]](t *testing.T, input []uint8, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint8 {
	t.Helper()

	vectors := VectorizeUint8[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint8, uint8](
			vectors,
			ro.Map(func(v V) []uint8 {
				var buffer [maxLanes]uint8
				n := v.StorePart(buffer[:])

				return append([]uint8{}, buffer[:n]...)
			}),
			ro.Flatten[uint8](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint8 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint8(mask simd.Mask8s) []bool {
	lanes := lanesUint8()

	var buf [maxLanes]int8
	mask.ToInt8s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func lanesUint16() int { return simd.BroadcastUint16s(0).Len() }

// sizeSweepUint16 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint16() []int {
	lanes := lanesUint16()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint16 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampUint16(size int) []uint16 {
	out := make([]uint16, size)
	for i := range out {
		out[i] = uint16(i%1000 + 1)
	}

	return out
}

// collectUint16Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint16s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint16Values[V Uint16Buffer[V]](t *testing.T, input []uint16, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint16 {
	t.Helper()

	vectors := VectorizeUint16[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint16, uint16](
			vectors,
			ro.Map(func(v V) []uint16 {
				var buffer [maxLanes]uint16
				n := v.StorePart(buffer[:])

				return append([]uint16{}, buffer[:n]...)
			}),
			ro.Flatten[uint16](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint16 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint16(mask simd.Mask16s) []bool {
	lanes := lanesUint16()

	var buf [maxLanes]int16
	mask.ToInt16s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

func lanesUint32() int { return simd.BroadcastUint32s(0).Len() }

// sizeSweepUint32 returns input lengths that surround every lane boundary, where the
// partial-vector path is most likely to be wrong.
func sizeSweepUint32() []int {
	lanes := lanesUint32()

	return []int{0, 1, 2, lanes - 1, lanes, lanes + 1, 2*lanes - 1, 2 * lanes, 2*lanes + 1, 35}
}

// rampUint32 builds 1, 2, 3... wrapping at 1000 so long inputs stay in range.
func rampUint32(size int) []uint32 {
	out := make([]uint32, size)
	for i := range out {
		out[i] = uint32(i%1000 + 1)
	}

	return out
}

// collectUint32Values runs a vector pipeline and flattens it back to scalars.
//
// It must be generic over V rather than naming PartialUint32s in its signature: a
// non-generic function holding a concrete simd-containing type in a parameter fails
// to compile, because the specializer clones the type and the synthesized dispatcher
// cannot convert the argument to the clone's type.
func collectUint32Values[V Uint32Buffer[V]](t *testing.T, input []uint32, operators ...func(ro.Observable[V]) ro.Observable[V]) []uint32 {
	t.Helper()

	vectors := VectorizeUint32[V]()(ro.FromSlice(input))
	for _, operator := range operators {
		vectors = operator(vectors)
	}

	values, err := ro.Collect(
		ro.Pipe2[V, []uint32, uint32](
			vectors,
			ro.Map(func(v V) []uint32 {
				var buffer [maxLanes]uint32
				n := v.StorePart(buffer[:])

				return append([]uint32{}, buffer[:n]...)
			}),
			ro.Flatten[uint32](),
		),
	)
	assert.NoError(t, err)

	return values
}

// maskLanesUint32 decodes a mask into one bool per lane, so a test can inspect it
// directly. Going through Select would prove nothing: Select re-applies the validity
// mask itself, and would hide a Contains that forgot to.
func maskLanesUint32(mask simd.Mask32s) []bool {
	lanes := lanesUint32()

	var buf [maxLanes]int32
	mask.ToInt32s().Store(buf[:lanes])

	out := make([]bool, lanes)
	for i := range out {
		out[i] = buf[i] != 0
	}

	return out
}

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

	vectors := VectorizeUint64[V]()(ro.FromSlice(input))
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

	vectors := VectorizeFloat32[V]()(ro.FromSlice(input))
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

	vectors := VectorizeFloat64[V]()(ro.FromSlice(input))
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
