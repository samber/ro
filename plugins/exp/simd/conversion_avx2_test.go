//go:build go1.26 && goexperiment.simd && amd64

package simd

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// TestScalarToInt8x32 tests AVX2 Int8x32 ScalarTo conversion
func TestScalarToInt8x32(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int8, 32)
	for i := 0; i < 32; i++ {
		input[i] = int8(i)
	}

	result, err := ro.Collect(ScalarToInt8x32[int8]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt8x32ToScalar tests AVX2 Int8x32 ToScalar conversion
func TestInt8x32ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int8, 32)
	for i := 0; i < 32; i++ {
		input[i] = int8(i)
	}

	vec := ScalarToInt8x32[int8]()(ro.Just(input...))
	result, err := ro.Collect(Int8x32ToScalar[int8]()(vec))

	is.NoError(err)
	is.Equal(32, len(result))
	for i := 0; i < 32; i++ {
		is.Equal(int8(i), result[i])
	}
}

// TestScalarToInt16x16 tests AVX2 Int16x16 ScalarTo conversion
func TestScalarToInt16x16(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int16, 16)
	for i := 0; i < 16; i++ {
		input[i] = int16(i)
	}

	result, err := ro.Collect(ScalarToInt16x16[int16]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt16x16ToScalar tests AVX2 Int16x16 ToScalar conversion
func TestInt16x16ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int16, 16)
	for i := 0; i < 16; i++ {
		input[i] = int16(i)
	}

	vec := ScalarToInt16x16[int16]()(ro.Just(input...))
	result, err := ro.Collect(Int16x16ToScalar[int16]()(vec))

	is.NoError(err)
	is.Equal(16, len(result))
	for i := 0; i < 16; i++ {
		is.Equal(int16(i), result[i])
	}
}

// TestScalarToInt32x8 tests AVX2 Int32x8 ScalarTo conversion
func TestScalarToInt32x8(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int32, 8)
	for i := 0; i < 8; i++ {
		input[i] = int32(i)
	}

	result, err := ro.Collect(ScalarToInt32x8[int32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt32x8ToScalar tests AVX2 Int32x8 ToScalar conversion
func TestInt32x8ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int32, 8)
	for i := 0; i < 8; i++ {
		input[i] = int32(i)
	}

	vec := ScalarToInt32x8[int32]()(ro.Just(input...))
	result, err := ro.Collect(Int32x8ToScalar[int32]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.Equal(int32(i), result[i])
	}
}

// TestScalarToInt64x4 tests AVX2 Int64x4 ScalarTo conversion
func TestScalarToInt64x4(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int64, 4)
	for i := 0; i < 4; i++ {
		input[i] = int64(i)
	}

	result, err := ro.Collect(ScalarToInt64x4[int64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt64x4ToScalar tests AVX2 Int64x4 ToScalar conversion
func TestInt64x4ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]int64, 4)
	for i := 0; i < 4; i++ {
		input[i] = int64(i)
	}

	vec := ScalarToInt64x4[int64]()(ro.Just(input...))
	result, err := ro.Collect(Int64x4ToScalar[int64]()(vec))

	is.NoError(err)
	is.Equal(4, len(result))
	for i := 0; i < 4; i++ {
		is.Equal(int64(i), result[i])
	}
}

// TestScalarToUint8x32 tests AVX2 Uint8x32 ScalarTo conversion
func TestScalarToUint8x32(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint8, 32)
	for i := 0; i < 32; i++ {
		input[i] = uint8(i)
	}

	result, err := ro.Collect(ScalarToUint8x32[uint8]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint8x32ToScalar tests AVX2 Uint8x32 ToScalar conversion
func TestUint8x32ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint8, 32)
	for i := 0; i < 32; i++ {
		input[i] = uint8(i)
	}

	vec := ScalarToUint8x32[uint8]()(ro.Just(input...))
	result, err := ro.Collect(Uint8x32ToScalar[uint8]()(vec))

	is.NoError(err)
	is.Equal(32, len(result))
	for i := 0; i < 32; i++ {
		is.Equal(uint8(i), result[i])
	}
}

// TestScalarToUint16x16 tests AVX2 Uint16x16 ScalarTo conversion
func TestScalarToUint16x16(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint16, 16)
	for i := 0; i < 16; i++ {
		input[i] = uint16(i)
	}

	result, err := ro.Collect(ScalarToUint16x16[uint16]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint16x16ToScalar tests AVX2 Uint16x16 ToScalar conversion
func TestUint16x16ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint16, 16)
	for i := 0; i < 16; i++ {
		input[i] = uint16(i)
	}

	vec := ScalarToUint16x16[uint16]()(ro.Just(input...))
	result, err := ro.Collect(Uint16x16ToScalar[uint16]()(vec))

	is.NoError(err)
	is.Equal(16, len(result))
	for i := 0; i < 16; i++ {
		is.Equal(uint16(i), result[i])
	}
}

// TestScalarToUint32x8 tests AVX2 Uint32x8 ScalarTo conversion
func TestScalarToUint32x8(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint32, 8)
	for i := 0; i < 8; i++ {
		input[i] = uint32(i)
	}

	result, err := ro.Collect(ScalarToUint32x8[uint32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint32x8ToScalar tests AVX2 Uint32x8 ToScalar conversion
func TestUint32x8ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint32, 8)
	for i := 0; i < 8; i++ {
		input[i] = uint32(i)
	}

	vec := ScalarToUint32x8[uint32]()(ro.Just(input...))
	result, err := ro.Collect(Uint32x8ToScalar[uint32]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.Equal(uint32(i), result[i])
	}
}

// TestScalarToUint64x4 tests AVX2 Uint64x4 ScalarTo conversion
func TestScalarToUint64x4(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint64, 4)
	for i := 0; i < 4; i++ {
		input[i] = uint64(i)
	}

	result, err := ro.Collect(ScalarToUint64x4[uint64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint64x4ToScalar tests AVX2 Uint64x4 ToScalar conversion
func TestUint64x4ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]uint64, 4)
	for i := 0; i < 4; i++ {
		input[i] = uint64(i)
	}

	vec := ScalarToUint64x4[uint64]()(ro.Just(input...))
	result, err := ro.Collect(Uint64x4ToScalar[uint64]()(vec))

	is.NoError(err)
	is.Equal(4, len(result))
	for i := 0; i < 4; i++ {
		is.Equal(uint64(i), result[i])
	}
}

// TestScalarToFloat32x8 tests AVX2 Float32x8 ScalarTo conversion
func TestScalarToFloat32x8(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]float32, 8)
	for i := 0; i < 8; i++ {
		input[i] = float32(i) + 0.5
	}

	result, err := ro.Collect(ScalarToFloat32x8[float32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestFloat32x8ToScalar tests AVX2 Float32x8 ToScalar conversion
func TestFloat32x8ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]float32, 8)
	for i := 0; i < 8; i++ {
		input[i] = float32(i) + 0.5
	}

	vec := ScalarToFloat32x8[float32]()(ro.Just(input...))
	result, err := ro.Collect(Float32x8ToScalar[float32]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.InDelta(float32(i)+0.5, result[i], 0.001)
	}
}

// TestScalarToFloat64x4 tests AVX2 Float64x4 ScalarTo conversion
func TestScalarToFloat64x4(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]float64, 4)
	for i := 0; i < 4; i++ {
		input[i] = float64(i) + 0.5
	}

	result, err := ro.Collect(ScalarToFloat64x4[float64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestFloat64x4ToScalar tests AVX2 Float64x4 ToScalar conversion
func TestFloat64x4ToScalar(t *testing.T) {
	requireAVX2(t)
	is := assert.New(t)

	input := make([]float64, 4)
	for i := 0; i < 4; i++ {
		input[i] = float64(i) + 0.5
	}

	vec := ScalarToFloat64x4[float64]()(ro.Just(input...))
	result, err := ro.Collect(Float64x4ToScalar[float64]()(vec))

	is.NoError(err)
	is.Equal(4, len(result))
	for i := 0; i < 4; i++ {
		is.InDelta(float64(i)+0.5, result[i], 0.001)
	}
}

// ==================== Round-trip Tests ====================

// TestRoundTripInt8x32 tests round-trip conversion for AVX2 Int8x32
func TestRoundTripInt8x32(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []int8
	}{
		{"empty", []int8{}},
		{"single", []int8{42}},
		{"partial", makeInt8Range(0, 15)},
		{"full buffer", makeInt8Range(0, 31)},
		{"multiple buffers", makeInt8Range(0, 63)},
		{"all max", makeInt8Filled(64, 127)},
		{"all min", makeInt8Filled(32, -128)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(ScalarToInt8x32[int8]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			// For partial buffers, result will include trailing zeros to fill the vector
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripInt16x16(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []int16
	}{
		{"empty", []int16{}},
		{"single", []int16{1000}},
		{"partial", makeInt16Range(0, 7)},
		{"full buffer", makeInt16Range(0, 15)},
		{"multiple buffers", makeInt16Range(0, 31)},
		{"large values", []int16{32767, 32766, -32768, -32767}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(ScalarToInt16x16[int16]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripInt32x8(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []int32
	}{
		{"empty", []int32{}},
		{"single", []int32{100000}},
		{"partial", makeInt32Range(0, 3)},
		{"full buffer", makeInt32Range(0, 7)},
		{"multiple buffers", makeInt32Range(0, 15)},
		{"max values", []int32{2147483647, 2147483646, -2147483648, -2147483647}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(ScalarToInt32x8[int32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripInt64x4(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []int64
	}{
		{"empty", []int64{}},
		{"single", []int64{1000000}},
		{"partial", makeInt64Range(0, 1)},
		{"full buffer", makeInt64Range(0, 3)},
		{"multiple buffers", makeInt64Range(0, 7)},
		{"max values", []int64{9223372036854775807, -9223372036854775808}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(ScalarToInt64x4[int64]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripUint8x32(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []uint8
	}{
		{"empty", []uint8{}},
		{"single", []uint8{42}},
		{"partial", makeUint8Filled(15, 1)},
		{"full buffer", makeUint8Filled(32, 255)},
		{"multiple buffers", makeUint8Filled(64, 128)},
		{"all max", makeUint8Filled(32, 255)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(ScalarToUint8x32[uint8]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripUint16x16(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []uint16
	}{
		{"empty", []uint16{}},
		{"single", []uint16{1000}},
		{"partial", makeUint16Filled(7, 100)},
		{"full buffer", makeUint16Filled(16, 65535)},
		{"multiple buffers", makeUint16Filled(32, 32768)},
		{"max values", []uint16{65535, 65534, 32768}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(ScalarToUint16x16[uint16]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripUint32x8(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []uint32
	}{
		{"empty", []uint32{}},
		{"single", []uint32{100000}},
		{"partial", makeUint32Filled(3, 1000)},
		{"full buffer", makeUint32Filled(8, 4294967295)},
		{"multiple buffers", makeUint32Filled(16, 2147483648)},
		{"max values", []uint32{4294967295, 4294967294, 2147483648}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(ScalarToUint32x8[uint32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripUint64x4(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []uint64
	}{
		{"empty", []uint64{}},
		{"single", []uint64{1000000}},
		{"partial", makeUint64Range(0, 1)},
		{"full buffer", makeUint64Range(0, 3)},
		{"multiple buffers", makeUint64Range(0, 7)},
		{"max values", []uint64{18446744073709551615, 18446744073709551614}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(ScalarToUint64x4[uint64]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i])
				}
			}
		})
	}
}

func TestRoundTripFloat32x8(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []float32
	}{
		{"empty", []float32{}},
		{"single", []float32{3.14}},
		{"partial", makeFloat32Range(0, 3)},
		{"full buffer", makeFloat32Range(0, 7)},
		{"multiple buffers", makeFloat32Range(0, 15)},
		{"negatives", []float32{-1.5, -2.5, -3.5, -4.5}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(ScalarToFloat32x8[float32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.InDelta(t, tc.input[i], result[i], 0.001)
				}
			}
		})
	}
}

func TestRoundTripFloat64x4(t *testing.T) {
	requireAVX2(t)
	testCases := []struct {
		name  string
		input []float64
	}{
		{"empty", []float64{}},
		{"single", []float64{3.14}},
		{"partial", []float64{1.1, 2.2}},
		{"full buffer", []float64{1.5, 2.5, 3.5, 4.5}},
		{"multiple buffers", []float64{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}},
		{"negatives", []float64{-1.5, -2.5, -3.5, -4.5}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(ScalarToFloat64x4[float64]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.True(t, len(result) >= len(tc.input))
			for i := range tc.input {
				if i < len(result) {
					assert.InDelta(t, tc.input[i], result[i], 0.001)
				}
			}
		})
	}
}
