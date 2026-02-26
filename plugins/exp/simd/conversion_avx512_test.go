//go:build go1.26 && goexperiment.simd && amd64

package simd

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// TestScalarToInt8x64 tests AVX-512 Int8x64 To conversion
func TestScalarToInt8x64(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int8, 64)
	for i := 0; i < 64; i++ {
		input[i] = int8(i)
	}

	result, err := ro.Collect(ScalarToInt8x64[int8]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt8x64ToScalar tests AVX-512 Int8x64 From conversion
func TestInt8x64ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int8, 64)
	for i := 0; i < 64; i++ {
		input[i] = int8(i)
	}

	vec := ScalarToInt8x64[int8]()(ro.Just(input...))
	result, err := ro.Collect(Int8x64ToScalar[int8]()(vec))

	is.NoError(err)
	is.Equal(64, len(result))
	for i := 0; i < 64; i++ {
		is.Equal(int8(i), result[i])
	}
}

// TestScalarToFloat32x16 tests AVX-512 Float32x16 To conversion
func TestScalarToFloat32x16(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]float32, 16)
	for i := 0; i < 16; i++ {
		input[i] = float32(i) + 0.5
	}

	result, err := ro.Collect(ScalarToFloat32x16[float32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestFloat32x16ToScalar tests AVX-512 Float32x16 From conversion
func TestFloat32x16ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]float32, 16)
	for i := 0; i < 16; i++ {
		input[i] = float32(i) + 0.5
	}

	vec := ScalarToFloat32x16[float32]()(ro.Just(input...))
	result, err := ro.Collect(Float32x16ToScalar[float32]()(vec))

	is.NoError(err)
	is.Equal(16, len(result))
	for i := 0; i < 16; i++ {
		is.InDelta(float32(i)+0.5, result[i], 0.001)
	}
}

// TestScalarToInt16x32 tests AVX-512 Int16x32 To conversion
func TestScalarToInt16x32(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int16, 32)
	for i := 0; i < 32; i++ {
		input[i] = int16(i)
	}

	result, err := ro.Collect(ScalarToInt16x32[int16]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt16x32ToScalar tests AVX-512 Int16x32 From conversion
func TestInt16x32ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int16, 32)
	for i := 0; i < 32; i++ {
		input[i] = int16(i)
	}

	vec := ScalarToInt16x32[int16]()(ro.Just(input...))
	result, err := ro.Collect(Int16x32ToScalar[int16]()(vec))

	is.NoError(err)
	is.Equal(32, len(result))
	for i := 0; i < 32; i++ {
		is.Equal(int16(i), result[i])
	}
}

// TestScalarToInt32x16 tests AVX-512 Int32x16 To conversion
func TestScalarToInt32x16(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int32, 16)
	for i := 0; i < 16; i++ {
		input[i] = int32(i)
	}

	result, err := ro.Collect(ScalarToInt32x16[int32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt32x16ToScalar tests AVX-512 Int32x16 From conversion
func TestInt32x16ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int32, 16)
	for i := 0; i < 16; i++ {
		input[i] = int32(i)
	}

	vec := ScalarToInt32x16[int32]()(ro.Just(input...))
	result, err := ro.Collect(Int32x16ToScalar[int32]()(vec))

	is.NoError(err)
	is.Equal(16, len(result))
	for i := 0; i < 16; i++ {
		is.Equal(int32(i), result[i])
	}
}

// TestScalarToInt64x8 tests AVX-512 Int64x8 To conversion
func TestScalarToInt64x8(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int64, 8)
	for i := 0; i < 8; i++ {
		input[i] = int64(i)
	}

	result, err := ro.Collect(ScalarToInt64x8[int64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestInt64x8ToScalar tests AVX-512 Int64x8 From conversion
func TestInt64x8ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int64, 8)
	for i := 0; i < 8; i++ {
		input[i] = int64(i)
	}

	vec := ScalarToInt64x8[int64]()(ro.Just(input...))
	result, err := ro.Collect(Int64x8ToScalar[int64]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.Equal(int64(i), result[i])
	}
}

// TestScalarToUint8x64 tests AVX-512 Uint8x64 To conversion
func TestScalarToUint8x64(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint8, 64)
	for i := 0; i < 64; i++ {
		input[i] = uint8(i)
	}

	result, err := ro.Collect(ScalarToUint8x64[uint8]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint8x64ToScalar tests AVX-512 Uint8x64 From conversion
func TestUint8x64ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint8, 64)
	for i := 0; i < 64; i++ {
		input[i] = uint8(i)
	}

	vec := ScalarToUint8x64[uint8]()(ro.Just(input...))
	result, err := ro.Collect(Uint8x64ToScalar[uint8]()(vec))

	is.NoError(err)
	is.Equal(64, len(result))
	for i := 0; i < 64; i++ {
		is.Equal(uint8(i), result[i])
	}
}

// TestScalarToUint16x32 tests AVX-512 Uint16x32 To conversion
func TestScalarToUint16x32(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint16, 32)
	for i := 0; i < 32; i++ {
		input[i] = uint16(i)
	}

	result, err := ro.Collect(ScalarToUint16x32[uint16]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint16x32ToScalar tests AVX-512 Uint16x32 From conversion
func TestUint16x32ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint16, 32)
	for i := 0; i < 32; i++ {
		input[i] = uint16(i)
	}

	vec := ScalarToUint16x32[uint16]()(ro.Just(input...))
	result, err := ro.Collect(Uint16x32ToScalar[uint16]()(vec))

	is.NoError(err)
	is.Equal(32, len(result))
	for i := 0; i < 32; i++ {
		is.Equal(uint16(i), result[i])
	}
}

// TestScalarToUint32x16 tests AVX-512 Uint32x16 To conversion
func TestScalarToUint32x16(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint32, 16)
	for i := 0; i < 16; i++ {
		input[i] = uint32(i)
	}

	result, err := ro.Collect(ScalarToUint32x16[uint32]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint32x16ToScalar tests AVX-512 Uint32x16 From conversion
func TestUint32x16ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint32, 16)
	for i := 0; i < 16; i++ {
		input[i] = uint32(i)
	}

	vec := ScalarToUint32x16[uint32]()(ro.Just(input...))
	result, err := ro.Collect(Uint32x16ToScalar[uint32]()(vec))

	is.NoError(err)
	is.Equal(16, len(result))
	for i := 0; i < 16; i++ {
		is.Equal(uint32(i), result[i])
	}
}

// TestScalarToUint64x8 tests AVX-512 Uint64x8 To conversion
func TestScalarToUint64x8(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint64, 8)
	for i := 0; i < 8; i++ {
		input[i] = uint64(i)
	}

	result, err := ro.Collect(ScalarToUint64x8[uint64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestUint64x8ToScalar tests AVX-512 Uint64x8 From conversion
func TestUint64x8ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]uint64, 8)
	for i := 0; i < 8; i++ {
		input[i] = uint64(i)
	}

	vec := ScalarToUint64x8[uint64]()(ro.Just(input...))
	result, err := ro.Collect(Uint64x8ToScalar[uint64]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.Equal(uint64(i), result[i])
	}
}

// TestScalarToFloat64x8 tests AVX-512 Float64x8 To conversion
func TestScalarToFloat64x8(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]float64, 8)
	for i := 0; i < 8; i++ {
		input[i] = float64(i) + 0.5
	}

	result, err := ro.Collect(ScalarToFloat64x8[float64]()(ro.Just(input...)))

	is.NoError(err)
	is.Equal(1, len(result))
}

// TestFloat64x8ToScalar tests AVX-512 Float64x8 From conversion
func TestFloat64x8ToScalar(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]float64, 8)
	for i := 0; i < 8; i++ {
		input[i] = float64(i) + 0.5
	}

	vec := ScalarToFloat64x8[float64]()(ro.Just(input...))
	result, err := ro.Collect(Float64x8ToScalar[float64]()(vec))

	is.NoError(err)
	is.Equal(8, len(result))
	for i := 0; i < 8; i++ {
		is.InDelta(float64(i)+0.5, result[i], 0.001)
	}
}

// TestRoundTripInt8x64 tests round-trip conversion for AVX-512 Int8x64
func TestRoundTripInt8x64(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]int8, 128)
	for i := 0; i < 128; i++ {
		input[i] = int8(i - 64)
	}

	result, err := ro.Collect(
		Int8x64ToScalar[int8]()(ScalarToInt8x64[int8]()(ro.Just(input...))),
	)

	is.NoError(err)
	is.Equal(128, len(result))
	for i := 0; i < 128; i++ {
		is.Equal(int8(i-64), result[i])
	}
}

// TestRoundTripFloat32x16 tests round-trip conversion for AVX-512 Float32x16
func TestRoundTripFloat32x16(t *testing.T) {
	requireAVX512(t)
	is := assert.New(t)

	input := make([]float32, 32)
	for i := 0; i < 32; i++ {
		input[i] = float32(i) + 0.5
	}

	result, err := ro.Collect(
		Float32x16ToScalar[float32]()(ScalarToFloat32x16[float32]()(ro.Just(input...))),
	)

	is.NoError(err)
	is.Equal(32, len(result))
	for i := 0; i < 32; i++ {
		is.InDelta(float32(i)+0.5, result[i], 0.001)
	}
}

// ==================== AVX-512 Round-trip Tests ====================

func TestRoundTripAVX512Int8x64(t *testing.T) {
	requireAVX512(t)

	pad64 := func(s []int8) []int8 {
		if len(s) == 0 {
			return s
		}
		n := ((len(s) + 63) / 64) * 64
		out := make([]int8, n)
		copy(out, s)
		return out
	}

	testCases := []struct {
		name     string
		input    []int8
		expected []int8
	}{
		{"empty", []int8{}, []int8{}},
		{"single", []int8{42}, pad64([]int8{42})},
		{"partial", makeInt8Range(0, 31), func() []int8 {
			out := make([]int8, 64)
			copy(out, makeInt8Range(0, 31))
			return out
		}()},
		{"full buffer", makeInt8Range(0, 63), makeInt8Range(0, 63)},
		{"multiple buffers", makeInt8Range(0, 127), makeInt8Range(0, 127)},
		{"all max", makeInt8Filled(128, 127), makeInt8Filled(128, 127)},
		{"all min", makeInt8Filled(64, -128), makeInt8Filled(64, -128)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(ScalarToInt8x64[int8]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result, "AVX-512 round-trip should preserve data (with trailing zeros for partial vectors)")
		})
	}
}

func TestRoundTripAVX512Float32x16(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name  string
		input []float32
	}{
		{"empty", []float32{}},
		{"full buffer", makeFloat32Range(0, 15)},
		{"multiple buffers", makeFloat32Range(0, 31)},
		{"negatives", makeFloat32Range(0, 15)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(ScalarToFloat32x16[float32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.input, result)
		})
	}
}
