//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"fmt"
	"simd/archsimd"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// ==================== ScalarToInt8x16 ====================

func TestScalarToInt8x16(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []int8
		expectedVectors int
		verifyContent   bool
		expectedValues  []int8 // first N values to verify
	}{
		{
			name:            "empty input",
			input:           []int8{},
			expectedVectors: 0,
		},
		{
			name:            "single value - partial",
			input:           []int8{5},
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{5},
		},
		{
			name:            "partial buffer (10 values)",
			input:           []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
		},
		{
			name:            "exactly one full buffer (16 values)",
			input:           []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:            "two full buffers (32 values)",
			input:           makeInt8Range(0, 31),
			expectedVectors: 2,
			verifyContent:   true,
			expectedValues:  []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:            "full buffer plus partial (20 values)",
			input:           makeInt8Range(0, 19),
			expectedVectors: 2,
			verifyContent:   true,
			expectedValues:  []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19},
		},
		{
			name:            "negative values",
			input:           []int8{-1, -2, -3, -4, -5},
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{-1, -2, -3, -4, -5},
		},
		{
			name:            "all max values (127)",
			input:           makeInt8Filled(16, 127),
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{127, 127, 127, 127},
		},
		{
			name:            "all min values (-128)",
			input:           makeInt8Filled(16, -128),
			expectedVectors: 1,
			verifyContent:   true,
			expectedValues:  []int8{-128, -128, -128, -128},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			result, err := ro.Collect(
				ScalarToInt8x16[int8]()(ro.Just(tc.input...)),
			)

			is.NoError(err)
			is.Equal(tc.expectedVectors, len(result), fmt.Sprintf("expected %d vectors, got %d", tc.expectedVectors, len(result)))

			if tc.verifyContent && len(result) > 0 {
				// Verify content by extracting values
				var extracted []int8
				for _, vec := range result {
					var buf [16]int8
					vec.Store(&buf)
					for i := 0; i < 16; i++ {
						if len(extracted) < len(tc.expectedValues) {
							extracted = append(extracted, buf[i])
						}
					}
				}
				is.Equal(tc.expectedValues, extracted, "vector content mismatch")
			}
		})
	}
}

// ==================== Round-trip Tests (ScalarTo -> ToScalar) ====================

func TestRoundTripInt8x16(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name  string
		input []int8
	}{
		{"empty", []int8{}},
		{"single", []int8{42}},
		{"full buffer", []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
		{"partial buffer", []int8{5, 10, 15, 20}},
		{"multiple buffers", makeInt8Range(0, 49)},
		{"all same value", makeInt8Filled(100, 42)},
		{"negative values", []int8{-1, -2, -3, -4, -5, -6, -7, -8}},
		{"alternating signs", []int8{-1, 1, -2, 2, -3, 3, -4, 4, -5, 5}},
		{"max/min boundary", []int8{127, 126, 125, -128, -127, -126}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(ScalarToInt8x16[int8]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			// For partial buffers, result will include trailing zeros to fill the vector
			// Check that input is a prefix of result
			assert.True(t, len(result) >= len(tc.input), "result should be at least as long as input")
			for i := range tc.input {
				if i < len(result) {
					assert.Equal(t, tc.input[i], result[i], "round-trip should preserve input values")
				}
			}
		})
	}
}

func TestRoundTripInt16x8(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name  string
		input []int16
	}{
		{"empty", []int16{}},
		{"single", []int16{1000}},
		{"full buffer", []int16{1, 2, 3, 4, 5, 6, 7, 8}},
		{"multiple buffers", makeInt16Range(0, 31)},
		{"large values", []int16{32767, 32766, -32768, -32767}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(ScalarToInt16x8[int16]()(ro.Just(tc.input...))),
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

func TestRoundTripInt32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name  string
		input []int32
	}{
		{"empty", []int32{}},
		{"single", []int32{100000}},
		{"full buffer", []int32{1, 2, 3, 4}},
		{"multiple buffers", makeInt32Range(0, 15)},
		{"max values", []int32{2147483647, 2147483646, -2147483648, -2147483647}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(ScalarToInt32x4[int32]()(ro.Just(tc.input...))),
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

func TestRoundTripInt64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name  string
		input []int64
	}{
		{"empty", []int64{}},
		{"single", []int64{1000000}},
		{"full buffer", []int64{1, 2}},
		{"multiple buffers", makeInt64Range(0, 9)},
		{"max values", []int64{9223372036854775807, -9223372036854775808}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Int64x2ToScalar[int64]()(ScalarToInt64x2[int64]()(ro.Just(tc.input...))),
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

// ==================== Int8x16ToScalar (Content Verification) ====================

func TestInt8x16ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []int8
		expected []int8
	}{
		{
			name:     "sequential 1-16",
			input:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			name:     "all zeros",
			expected: []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:     "all max values (127)",
			expected: []int8{127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
		},
		{
			name:     "all min values (-128)",
			expected: []int8{-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128},
		},
		{
			name:     "alternating 1 and -1",
			expected: []int8{-1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1},
		},
		{
			name:     "sequential -8 to 7",
			expected: []int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			// Create input vector for test
			var input []int8
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := archsimd.Int8x16{}
			for i, v := range input {
				vec = vec.SetElem(uint8(i), v)
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// ==================== Float32 Round-trip ====================

func TestRoundTripFloat32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{"empty", []float32{}, []float32{}},
		{"single", []float32{3.14}, []float32{3.14, 0, 0, 0}},
		{"full buffer", []float32{1.5, 2.5, 3.5, 4.5}, []float32{1.5, 2.5, 3.5, 4.5}},
		{"partial", []float32{1.5, 2.5, 3.5}, []float32{1.5, 2.5, 3.5, 0}},
		{"multiple buffers", makeFloat32Range(0, 11), makeFloat32Range(0, 11)},
		{"negatives", []float32{-1.5, -2.5, -3.5, -4.5}, []float32{-1.5, -2.5, -3.5, -4.5}},
		{"mixed signs", []float32{-1.5, 2.5, -3.5, 4.5}, []float32{-1.5, 2.5, -3.5, 4.5}},
		{"small values", []float32{0.1, 0.2, 0.3, 0.4}, []float32{0.1, 0.2, 0.3, 0.4}},
		{"large values", []float32{1e10, 2e10, 3e10, 4e10}, []float32{1e10, 2e10, 3e10, 4e10}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(ScalarToFloat32x4[float32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestRoundTripFloat64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{"empty", []float64{}, []float64{}},
		{"single", []float64{3.14159265359}, []float64{3.14159265359, 0}},
		{"full buffer", []float64{1.5, 2.5}, []float64{1.5, 2.5}},
		{"multiple buffers", []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}},
		{"negatives", []float64{-1.5, -2.5}, []float64{-1.5, -2.5}},
		{"very large", []float64{1e100, 2e100}, []float64{1e100, 2e100}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(ScalarToFloat64x2[float64]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// ==================== Uint Type Round-trip Tests ====================

func TestRoundTripUint8x16(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint8
		expected []uint8
	}{
		{"empty", []uint8{}, []uint8{}},
		{"single", []uint8{255}, []uint8{255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		{"full buffer", []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}},
		{"all max", makeUint8Filled(16, 255), makeUint8Filled(16, 255)},
		{"mixed", []uint8{0, 128, 255, 100}, []uint8{0, 128, 255, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(ScalarToUint8x16[uint8]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestRoundTripUint16x8(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint16
		expected []uint16
	}{
		{"empty", []uint16{}, []uint16{}},
		{"single", []uint16{65535}, []uint16{65535, 0, 0, 0, 0, 0, 0, 0}},
		{"full buffer", []uint16{1, 2, 3, 4, 5, 6, 7, 8}, []uint16{1, 2, 3, 4, 5, 6, 7, 8}},
		{"all max", makeUint16Filled(8, 65535), makeUint16Filled(8, 65535)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(ScalarToUint16x8[uint16]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestRoundTripUint32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint32
		expected []uint32
	}{
		{"empty", []uint32{}, []uint32{}},
		{"single", []uint32{4294967295}, []uint32{4294967295, 0, 0, 0}},
		{"full buffer", []uint32{1, 2, 3, 4}, []uint32{1, 2, 3, 4}},
		{"all max", makeUint32Filled(4, 4294967295), makeUint32Filled(4, 4294967295)},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(ScalarToUint32x4[uint32]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestRoundTripUint64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint64
		expected []uint64
	}{
		{"empty", []uint64{}, []uint64{}},
		{"single", []uint64{18446744073709551615}, []uint64{18446744073709551615, 0}},
		{"full buffer", []uint64{1, 2}, []uint64{1, 2}},
		{"all max", []uint64{18446744073709551615, 18446744073709551615}, []uint64{18446744073709551615, 18446744073709551615}},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				Uint64x2ToScalar[uint64]()(ScalarToUint64x2[uint64]()(ro.Just(tc.input...))),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// ==================== Individual ScalarTo Tests ====================

func TestScalarToInt16x8(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []int16
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []int16{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []int16{42},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (5 values)",
			input:           []int16{1, 2, 3, 4, 5},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (8 values)",
			input:           []int16{1, 2, 3, 4, 5, 6, 7, 8},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (16 values)",
			input:           makeInt16Range(0, 15),
			expectedVectors: 2,
		},
		{
			name:            "full buffer plus partial (10 values)",
			input:           makeInt16Range(0, 9),
			expectedVectors: 2,
		},
		{
			name:            "negative values",
			input:           []int16{-1, -2, -3, -4},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToInt16x8[int16]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToInt32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []int32
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []int32{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []int32{42},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (3 values)",
			input:           []int32{1, 2, 3},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (4 values)",
			input:           []int32{1, 2, 3, 4},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (8 values)",
			input:           makeInt32Range(0, 7),
			expectedVectors: 2,
		},
		{
			name:            "full buffer plus partial (6 values)",
			input:           makeInt32Range(0, 5),
			expectedVectors: 2,
		},
		{
			name:            "negative values",
			input:           []int32{-1, -2, -3, -4},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToInt32x4[int32]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToInt64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []int64
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []int64{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []int64{42},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (2 values)",
			input:           []int64{1, 2},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (4 values)",
			input:           makeInt64Range(0, 3),
			expectedVectors: 2,
		},
		{
			name:            "full buffer plus partial (3 values)",
			input:           makeInt64Range(0, 2),
			expectedVectors: 2,
		},
		{
			name:            "negative values",
			input:           []int64{-1, -2},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToInt64x2[int64]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToFloat32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []float32
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []float32{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []float32{3.14},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (3 values)",
			input:           []float32{1.5, 2.5, 3.5},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (4 values)",
			input:           []float32{1.5, 2.5, 3.5, 4.5},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (8 values)",
			input:           makeFloat32Range(0, 7),
			expectedVectors: 2,
		},
		{
			name:            "negative values",
			input:           []float32{-1.5, -2.5, -3.5, -4.5},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToFloat32x4[float32]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToFloat64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []float64
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []float64{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []float64{3.14159},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (2 values)",
			input:           []float64{1.5, 2.5},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (4 values)",
			input:           []float64{1.0, 2.0, 3.0, 4.0},
			expectedVectors: 2,
		},
		{
			name:            "negative values",
			input:           []float64{-1.5, -2.5},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToFloat64x2[float64]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToUint8x16(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []uint8
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []uint8{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []uint8{255},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (10 values)",
			input:           []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (16 values)",
			input:           []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (32 values)",
			input:           makeUint8Filled(32, 42),
			expectedVectors: 2,
		},
		{
			name:            "all max values",
			input:           makeUint8Filled(16, 255),
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToUint8x16[uint8]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToUint16x8(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []uint16
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []uint16{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []uint16{65535},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (5 values)",
			input:           []uint16{1, 2, 3, 4, 5},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (8 values)",
			input:           []uint16{1, 2, 3, 4, 5, 6, 7, 8},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (16 values)",
			input:           makeUint16Filled(16, 1000),
			expectedVectors: 2,
		},
		{
			name:            "all max values",
			input:           makeUint16Filled(8, 65535),
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToUint16x8[uint16]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToUint32x4(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []uint32
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []uint32{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []uint32{4294967295},
			expectedVectors: 1,
		},
		{
			name:            "partial buffer (3 values)",
			input:           []uint32{1, 2, 3},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (4 values)",
			input:           []uint32{1, 2, 3, 4},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (8 values)",
			input:           makeUint32Filled(8, 100000),
			expectedVectors: 2,
		},
		{
			name:            "all max values",
			input:           makeUint32Filled(4, 4294967295),
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToUint32x4[uint32]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

func TestScalarToUint64x2(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name            string
		input           []uint64
		expectedVectors int
	}{
		{
			name:            "empty input",
			input:           []uint64{},
			expectedVectors: 0,
		},
		{
			name:            "single value",
			input:           []uint64{18446744073709551615},
			expectedVectors: 1,
		},
		{
			name:            "exactly one full buffer (2 values)",
			input:           []uint64{1, 2},
			expectedVectors: 1,
		},
		{
			name:            "two full buffers (4 values)",
			input:           makeUint64Range(0, 3),
			expectedVectors: 2,
		},
		{
			name:            "all max values",
			input:           []uint64{18446744073709551615, 18446744073709551615},
			expectedVectors: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			result, err := ro.Collect(
				ScalarToUint64x2[uint64]()(ro.Just(tc.input...)),
			)

			assert.NoError(t, err)
			assert.Equal(t, tc.expectedVectors, len(result))
		})
	}
}

// ==================== Individual ToScalar Tests ====================

func TestInt16x8ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []int16
		expected []int16
	}{
		{
			name:     "sequential 1-8",
			input:    []int16{1, 2, 3, 4, 5, 6, 7, 8},
			expected: []int16{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "all zeros",
			expected: []int16{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []int16{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:     "all max values",
			expected: []int16{32767, 32767, 32767, 32767, 32767, 32767, 32767, 32767},
		},
		{
			name:     "all min values",
			expected: []int16{-32768, -32768, -32768, -32768, -32768, -32768, -32768, -32768},
		},
		{
			name:     "alternating 1 and -1",
			expected: []int16{-1, 1, -1, 1, -1, 1, -1, 1},
		},
		{
			name:     "sequential -4 to 3",
			expected: []int16{-4, -3, -2, -1, 0, 1, 2, 3},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []int16
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToInt16x8[int16]()(ro.Just(input...))
			result, err := ro.Collect(Int16x8ToScalar[int16]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestInt32x4ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []int32
		expected []int32
	}{
		{
			name:     "sequential 1-4",
			input:    []int32{1, 2, 3, 4},
			expected: []int32{1, 2, 3, 4},
		},
		{
			name:     "all zeros",
			expected: []int32{0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []int32{1, 1, 1, 1},
		},
		{
			name:     "sequential -2 to 1",
			expected: []int32{-2, -1, 0, 1},
		},
		{
			name:     "large values",
			expected: []int32{1000000, 2000000, 3000000, 4000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []int32
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToInt32x4[int32]()(ro.Just(input...))
			result, err := ro.Collect(Int32x4ToScalar[int32]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestInt64x2ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []int64
		expected []int64
	}{
		{
			name:     "sequential 1-2",
			input:    []int64{1, 2},
			expected: []int64{1, 2},
		},
		{
			name:     "all zeros",
			expected: []int64{0, 0},
		},
		{
			name:     "all ones",
			expected: []int64{1, 1},
		},
		{
			name:     "sequential -1 to 0",
			expected: []int64{-1, 0},
		},
		{
			name:     "large values",
			expected: []int64{1000000000, 2000000000},
		},
		{
			name:     "max values",
			expected: []int64{9223372036854775807, -9223372036854775808},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []int64
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToInt64x2[int64]()(ro.Just(input...))
			result, err := ro.Collect(Int64x2ToScalar[int64]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestFloat32x4ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []float32
		expected []float32
	}{
		{
			name:     "sequential 1.5-4.5",
			input:    []float32{1.5, 2.5, 3.5, 4.5},
			expected: []float32{1.5, 2.5, 3.5, 4.5},
		},
		{
			name:     "all zeros",
			expected: []float32{0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []float32{1, 1, 1, 1},
		},
		{
			name:     "negative values",
			expected: []float32{-1.5, -2.5, -3.5, -4.5},
		},
		{
			name:     "mixed signs",
			expected: []float32{-1.5, 2.5, -3.5, 4.5},
		},
		{
			name:     "small values",
			expected: []float32{0.1, 0.2, 0.3, 0.4},
		},
		{
			name:     "large values",
			expected: []float32{1e10, 2e10, 3e10, 4e10},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []float32
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToFloat32x4[float32]()(ro.Just(input...))
			result, err := ro.Collect(Float32x4ToScalar[float32]()(vec))

			assert.NoError(t, err)
			for i := range result {
				assert.InDelta(t, tc.expected[i], result[i], 0.0001)
			}
		})
	}
}

func TestFloat64x2ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []float64
		expected []float64
	}{
		{
			name:     "sequential 1.5-2.5",
			input:    []float64{1.5, 2.5},
			expected: []float64{1.5, 2.5},
		},
		{
			name:     "all zeros",
			expected: []float64{0, 0},
		},
		{
			name:     "all ones",
			expected: []float64{1, 1},
		},
		{
			name:     "negative values",
			expected: []float64{-1.5, -2.5},
		},
		{
			name:     "very large",
			expected: []float64{1e100, 2e100},
		},
		{
			name:     "pi and e",
			expected: []float64{3.14159265359, 2.71828182846},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []float64
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToFloat64x2[float64]()(ro.Just(input...))
			result, err := ro.Collect(Float64x2ToScalar[float64]()(vec))

			assert.NoError(t, err)
			for i := range result {
				assert.InDelta(t, tc.expected[i], result[i], 0.0000001)
			}
		})
	}
}

func TestUint8x16ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint8
		expected []uint8
	}{
		{
			name:     "sequential 0-15",
			input:    []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expected: []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:     "all zeros",
			expected: []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "all max values (255)",
			expected: []uint8{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
		},
		{
			name:     "alternating 0 and 255",
			expected: []uint8{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0},
		},
		{
			name:     "sequential 120-135 (wrapped)",
			expected: []uint8{120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []uint8
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToUint8x16[uint8]()(ro.Just(input...))
			result, err := ro.Collect(Uint8x16ToScalar[uint8]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestUint16x8ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint16
		expected []uint16
	}{
		{
			name:     "sequential 1-8",
			input:    []uint16{1, 2, 3, 4, 5, 6, 7, 8},
			expected: []uint16{1, 2, 3, 4, 5, 6, 7, 8},
		},
		{
			name:     "all zeros",
			expected: []uint16{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []uint16{1, 1, 1, 1, 1, 1, 1, 1},
		},
		{
			name:     "all max values (65535)",
			expected: []uint16{65535, 65535, 65535, 65535, 65535, 65535, 65535, 65535},
		},
		{
			name:     "sequential 32764-32771 (wrapped)",
			expected: []uint16{32764, 32765, 32766, 32767, 32768, 32769, 32770, 32771},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []uint16
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToUint16x8[uint16]()(ro.Just(input...))
			result, err := ro.Collect(Uint16x8ToScalar[uint16]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestUint32x4ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint32
		expected []uint32
	}{
		{
			name:     "sequential 1-4",
			input:    []uint32{1, 2, 3, 4},
			expected: []uint32{1, 2, 3, 4},
		},
		{
			name:     "all zeros",
			expected: []uint32{0, 0, 0, 0},
		},
		{
			name:     "all ones",
			expected: []uint32{1, 1, 1, 1},
		},
		{
			name:     "large values",
			expected: []uint32{1000000, 2000000, 3000000, 4000000},
		},
		{
			name:     "max values",
			expected: []uint32{4294967295, 4294967295, 4294967295, 4294967295},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []uint32
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToUint32x4[uint32]()(ro.Just(input...))
			result, err := ro.Collect(Uint32x4ToScalar[uint32]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}

func TestUint64x2ToScalar(t *testing.T) {
	requireAVX(t)
	testCases := []struct {
		name     string
		input    []uint64
		expected []uint64
	}{
		{
			name:     "sequential 1-2",
			input:    []uint64{1, 2},
			expected: []uint64{1, 2},
		},
		{
			name:     "all zeros",
			expected: []uint64{0, 0},
		},
		{
			name:     "all ones",
			expected: []uint64{1, 1},
		},
		{
			name:     "large values",
			expected: []uint64{1000000000, 2000000000},
		},
		{
			name:     "max values",
			expected: []uint64{18446744073709551615, 18446744073709551615},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			// Create input vector for test
			var input []uint64
			if tc.input != nil {
				input = tc.input
			} else {
				input = tc.expected
			}

			vec := ScalarToUint64x2[uint64]()(ro.Just(input...))
			result, err := ro.Collect(Uint64x2ToScalar[uint64]()(vec))

			assert.NoError(t, err)
			assert.Equal(t, tc.expected, result)
		})
	}
}
