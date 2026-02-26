//go:build go1.26 && goexperiment.simd && amd64

package simd

import (
	"math"
	"testing"

	"simd/archsimd"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// ==================== ReduceSumInt8x64 ====================

func TestReduceSumInt8x64(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    any
		expected int8
		wantErr  bool
	}{
		// Edge cases - empty and single
		{
			name:     "empty input",
			input:    []int8{},
			expected: 0,
		},
		{
			name:     "single value",
			input:    []int8{5},
			expected: 5,
		},
		{
			name:     "single negative value",
			input:    []int8{-10},
			expected: -10,
		},
		{
			name:     "single zero",
			input:    []int8{0},
			expected: 0,
		},
		{
			name:     "single max int8",
			input:    []int8{127},
			expected: 127,
		},
		{
			name:     "single min int8",
			input:    []int8{-128},
			expected: -128,
		},

		// Small inputs (partial buffer)
		{
			name:     "two values positive",
			input:    []int8{10, 20},
			expected: 30,
		},
		{
			name:     "two values negative",
			input:    []int8{-10, -20},
			expected: -30,
		},
		{
			name:     "two values mixed",
			input:    []int8{10, -20},
			expected: -10,
		},
		{
			name:     "small buffer (5 values)",
			input:    []int8{1, 2, 3, 4, 5},
			expected: 15,
		},
		{
			name:     "small buffer with zeros",
			input:    []int8{10, 0, -10, 0, 5},
			expected: 5,
		},

		// Partial buffer (less than 64)
		{
			name: "partial buffer (16 values)",
			input: func() []int8 {
				v := make([]int8, 16)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 16,
		},
		{
			name: "partial buffer (32 values)",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 32,
		},
		{
			name: "partial buffer (48 values)",
			input: func() []int8 {
				v := make([]int8, 48)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 48,
		},
		{
			name: "partial buffer (63 values)",
			input: func() []int8 {
				v := make([]int8, 63)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 63,
		},

		// Full buffer (64 values)
		{
			name: "single vector full (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 64,
		},
		{
			name: "full buffer with twos",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 2
				}
				return v
			}(),
			expected: -128, // 2*64 = 128 wraps to -128
		},
		{
			name: "full buffer with negative",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -1
				}
				return v
			}(),
			expected: -64,
		},

		// Multiple buffers (overflow)
		{
			name: "all ones (128 values)",
			input: func() []int8 {
				v := make([]int8, 128)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: -128, // wraps to negative
		},
		{
			name: "all ones (192 values)",
			input: func() []int8 {
				v := make([]int8, 192)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: -64, // 192 wraps to -64 (256 - 64)
		},
		{
			name: "all ones (256 values)",
			input: func() []int8 {
				v := make([]int8, 256)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 0, // wraps to 0
		},

		// Negative values
		{
			name: "all negative small (16 values)",
			input: func() []int8 {
				v := make([]int8, 16)
				for i := range v {
					v[i] = -1
				}
				return v
			}(),
			expected: -16,
		},
		{
			name: "all negative large (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -2
				}
				return v
			}(),
			expected: -128, // -2*64 = -128
		},
		{
			name: "alternating 1 and -1 (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 1
					} else {
						v[i] = -1
					}
				}
				return v
			}(),
			expected: 0,
		},

		// Mixed positive and negative
		{
			name:     "mixed small",
			input:    []int8{10, -5, 3, -2, 7, -3},
			expected: 10,
		},
		{
			name: "mixed large (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 5
					} else {
						v[i] = -3
					}
				}
				return v
			}(),
			expected: 64, // (5-3)*32 = 2*32 = 64
		},

		// Type alias and error propagation
		{
			name:     "custom int8 type alias",
			input:    []myInt8{1, 2, 3, 4, 5},
			expected: 15,
		},
		{
			name:     "error propagation",
			input:    ro.Throw[int8](assert.AnError),
			expected: 0,
			wantErr:  true,
		},

		// Zero handling
		{
			name: "all zeros (32 values)",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
			expected: 0,
		},
		{
			name: "zeros with values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 10
					} else {
						v[i] = 0
					}
				}
				return v
			}(),
			expected: -128, // 10*32 = 320 wraps to 64
		},

		// Boundary values
		{
			name: "max values (127)",
			input: func() []int8 {
				v := make([]int8, 4)
				for i := range v {
					v[i] = 127
				}
				return v
			}(),
			expected: -4, // 127*4 = 508, 508-512 = -4
		},
		{
			name: "min values (-128)",
			input: func() []int8 {
				v := make([]int8, 4)
				for i := range v {
					v[i] = -128
				}
				return v
			}(),
			expected: 0, // -128*4 = -512 wraps to 0
		},
		{
			name: "alternating max and min (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			expected: -32, // (127-128)*32 = -32
		},

		// Overflow/underflow scenarios
		{
			name: "overflow wraps (100)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			expected: -96, // 100*64 = 6400, 6400 % 256 = 160, 160-256 = -96
		},
		{
			name: "underflow wraps (-100)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			expected: 96, // -100*64 = -6400, -6400 % 256 = 96
		},

		// Large inputs
		{
			name: "large input (200 values)",
			input: func() []int8 {
				v := make([]int8, 200)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: -56, // 200 % 256 = 200, 200-256 = -56
		},
		{
			name: "large input (300 values)",
			input: func() []int8 {
				v := make([]int8, 300)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 44, // 300 % 256 = 44
		},
		{
			name: "large input (512 values)",
			input: func() []int8 {
				v := make([]int8, 512)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 0, // 512 % 256 = 0
		},

		// Sequential patterns
		{
			name: "sequence 1-16",
			input: func() []int8 {
				v := make([]int8, 16)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			expected: -120, // sum of 1-16 = 136 wraps to -120 in int8
		},
		{
			name: "sequence repeating (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i%16 + 1)
				}
				return v
			}(),
			expected: 32, // sum of 1-16 = 136, repeated 4 times = 544, 544-512 = 32
		},

		// Special patterns
		{
			name: "powers of 2",
			input: func() []int8 {
				v := make([]int8, 8)
				for i := range v {
					v[i] = int8(1 << i)
				}
				return v
			}(),
			expected: -1, // 1+2+4+8+16+32+64+128 = 255 = -1 in int8
		},
		{
			name: "decrementing from 10",
			input: func() []int8 {
				v := make([]int8, 10)
				for i := range v {
					v[i] = int8(10 - i)
				}
				return v
			}(),
			expected: 55, // sum of 10+9+...+1 = 55
		},

		// All same values
		{
			name: "all fives (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			expected: 64, // 5*64 = 320, 320-256 = 64
		},
		{
			name: "all tens (64 values)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			expected: -128, // 10*64 = 640, 640 wraps to -128 in int8
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int8
			var err error

			switch v := tc.input.(type) {
			case []int8:
				result, err = ro.Collect(
					ReduceSumInt8x64[int8]()(ScalarToInt8x64[int8]()(ro.Just(v...))),
				)
			case []myInt8:
				customResult, customErr := ro.Collect(
					ReduceSumInt8x64[myInt8]()(ScalarToInt8x64[myInt8]()(ro.Just(v...))),
				)
				err = customErr
				result = make([]int8, len(customResult))
				for i, val := range customResult {
					result[i] = int8(val)
				}
			case ro.Observable[int8]:
				result, err = ro.Collect(
					ReduceSumInt8x64[int8]()(ScalarToInt8x64[int8]()(v)),
				)
			}

			if tc.wantErr {
				is.Error(err)
				is.Empty(result)
			} else {
				is.NoError(err)
				is.Equal(tc.expected, result[0], "sum should match expected")
			}
		})
	}
}

// ==================== ReduceSumFloat32x16 ====================

func TestReduceSumFloat32x16(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name:     "empty input",
			input:    []float32{},
			expected: 0,
		},
		{
			name:     "single value",
			input:    []float32{3.14},
			expected: 3.14,
		},
		{
			name:     "full buffer (16 values)",
			input:    []float32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: 136,
		},
		{
			name: "overflow buffer (20 values)",
			input: func() []float32 {
				v := make([]float32, 20)
				for i := range v {
					v[i] = float32(i + 1)
				}
				return v
			}(),
			expected: 210,
		},
		{
			name: "all ones (32 values)",
			input: func() []float32 {
				v := make([]float32, 32)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 32,
		},
		{
			name: "negative values",
			input: []float32{-1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5,
				-9.5, -10.5, -11.5, -12.5},
			expected: -84,
		},
		{
			name: "zero values",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
			expected: 0,
		},
		{
			name:     "fractional values",
			input:    []float32{0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5},
			expected: 32,
		},
		{
			name:     "mixed positive and negative",
			input:    []float32{10, -5, 3, -2, 7, -3, 1, -1},
			expected: 10,
		},
		{
			name:     "large values",
			input:    []float32{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			expected: 36000000,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: float32(math.NaN()),
		},
		{
			name:     "positive Inf",
			input:    []float32{1, 2, float32(math.Inf(1)), 4},
			expected: float32(math.Inf(1)),
		},
		{
			name:     "negative Inf",
			input:    []float32{float32(math.Inf(-1)), 2, 3, 4},
			expected: float32(math.Inf(-1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			result, err := ro.Collect(
				ReduceSumFloat32x16[float32]()(ScalarToFloat32x16[float32]()(ro.Just(tc.input...))),
			)

			is.NoError(err)
			if math.IsNaN(float64(tc.expected)) {
				is.True(math.IsNaN(float64(result[0])))
			} else if math.IsInf(float64(tc.expected), 0) {
				is.Equal(math.IsInf(float64(tc.expected), 1), math.IsInf(float64(result[0]), 1))
				is.Equal(math.IsInf(float64(tc.expected), -1), math.IsInf(float64(result[0]), -1))
			} else {
				is.InDelta(tc.expected, result[0], 0.001)
			}
		})
	}
}

// ==================== Int8x64 Clamp/Min/Max tests ====================

func TestClampInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int8
		min       int8
		max       int8
		expected  []int8
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			min: -10,
			max: 10,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := 0; i < 64; i++ {
					val := i - 32
					if val < -10 {
						v[i] = -10
					} else if val > 10 {
						v[i] = 10
					} else {
						v[i] = int8(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			min: -50,
			max: 50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -50
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			min: -50,
			max: 50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "min equals max",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			min: 0,
			max: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []int8{},
			min:       100,
			max:       50,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampInt8x64[int8](tc.min, tc.max)(ro.Empty[*archsimd.Int8x64]())
				})
				return
			}

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(ClampInt8x64[int8](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		expected int8
	}{
		{
			name: "mixed values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			expected: -32,
		},
		{
			name: "all positive - detects accumulator init bug",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			expected: 1,
		},
		{
			name: "all negative",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -int8(i + 1)
				}
				return v
			}(),
			expected: -64,
		},
		{
			name: "boundary values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			expected: -128,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt8x64[int8]()(vec))

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		expected int8
	}{
		{
			name: "mixed values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			expected: 31,
		},
		{
			name: "all negative - detects accumulator init bug",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -int8(i + 1)
				}
				return v
			}(),
			expected: -1,
		},
		{
			name: "all positive",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			expected: 64,
		},
		{
			name: "boundary values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			expected: 127,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt8x64[int8]()(vec))

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

func TestMinInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int8
		threshold int8
		expected  []int8
	}{
		{
			name: "basic min",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := 0; i < 64; i++ {
					val := i - 32
					if val < 0 {
						v[i] = 0
					} else {
						v[i] = int8(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(MinInt8x64[int8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int8
		threshold int8
		expected  []int8
	}{
		{
			name: "basic max",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := 0; i < 64; i++ {
					val := i - 32
					if val > 0 {
						v[i] = 0
					} else {
						v[i] = int8(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(MaxInt8x64[int8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestAddInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		addend   int8
		expected []int8
	}{
		{
			name: "basic addition",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			addend: 5,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 15
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -106
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			addend: -50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 106
				}
				return v
			}(),
		},
		{
			name: "all zeros",
			input: func() []int8 {
				v := make([]int8, 64)
				return v
			}(),
			addend: 5,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
		},
		{
			name: "boundary values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			addend: 1,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = -128
					} else {
						v[i] = -127
					}
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(AddInt8x64[int8](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []int8
		subtrahend int8
		expected   []int8
	}{
		{
			name: "basic subtraction",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			subtrahend: 10,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 40
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = int8(i - 32)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			subtrahend: 50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -40
				}
				return v
			}(),
		},
		{
			name: "overflow (negative - negative)",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			subtrahend: -50,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					v[i] = -50
				}
				return v
			}(),
		},
		{
			name: "boundary values",
			input: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = -128
					} else {
						v[i] = 127
					}
				}
				return v
			}(),
			subtrahend: 1,
			expected: func() []int8 {
				v := make([]int8, 64)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = 126
					}
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt8x64[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x64ToScalar[int8]()(SubInt8x64[int8](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// ==================== Float32x16 tests ====================

func TestAddFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		addend   float32
		expected []float32
	}{
		{
			name: "basic addition",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 1.5
				}
				return v
			}(),
			addend: 2.5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 4.0
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
			addend: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
		},
		{
			name: "negative values",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -10.5
				}
				return v
			}(),
			addend: 5.5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -5.0
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = 10.0
					} else {
						v[i] = -10.0
					}
				}
				return v
			}(),
			addend: 5.0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = 15.0
					} else {
						v[i] = -5.0
					}
				}
				return v
			}(),
		},
		{
			name: "large values",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 1000000.0
				}
				return v
			}(),
			addend: 500000.0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 1500000.0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(AddFloat32x16[float32](tc.addend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []float32
		subtrahend float32
		expected   []float32
	}{
		{
			name: "basic subtraction",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 5.5
				}
				return v
			}(),
			subtrahend: 2.5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 3.0
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
		},
		{
			name: "negative result",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 5.0
				}
				return v
			}(),
			subtrahend: 10.0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -5.0
				}
				return v
			}(),
		},
		{
			name: "negative values",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -10.0
				}
				return v
			}(),
			subtrahend: 5.0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -15.0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(SubFloat32x16[float32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float32
		min       float32
		max       float32
		expected  []float32
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i - 8)
				}
				return v
			}(),
			min: -5,
			max: 5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := 0; i < 16; i++ {
					val := float32(i - 8)
					if val < -5 {
						v[i] = -5
					} else if val > 5 {
						v[i] = 5
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -10
				}
				return v
			}(),
			min: -5,
			max: 5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -5
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			min: -5,
			max: 5,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []float32{},
			min:       10,
			max:       5,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampFloat32x16[float32](tc.min, tc.max)(ro.Empty[*archsimd.Float32x16]())
				})
				return
			}

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(ClampFloat32x16[float32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float32
		threshold float32
		expected  []float32
	}{
		{
			name: "basic min",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i - 8)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := 0; i < 16; i++ {
					val := float32(i - 8)
					if val < 0 {
						v[i] = 0
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -10
				}
				return v
			}(),
			threshold: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(MinFloat32x16[float32](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float32
		threshold float32
		expected  []float32
	}{
		{
			name: "basic max",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i - 8)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := 0; i < 16; i++ {
					val := float32(i - 8)
					if val > 0 {
						v[i] = 0
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x16ToScalar[float32]()(MaxFloat32x16[float32](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceMinFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name: "sequential -8 to 7",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i - 8)
				}
				return v
			}(),
			expected: -8,
		},
		{
			name: "all same",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = 5.5
				}
				return v
			}(),
			expected: 5.5,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: float32(math.NaN()),
		},
		{
			name:     "negative Inf is min",
			input:    []float32{1, 2, float32(math.Inf(-1)), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: float32(math.Inf(-1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinFloat32x16[float32]()(vec))

			is.NoError(err)
			if math.IsNaN(float64(tc.expected)) {
				is.True(math.IsNaN(float64(result[0])))
			} else if math.IsInf(float64(tc.expected), -1) {
				is.True(math.IsInf(float64(result[0]), -1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

func TestReduceMaxFloat32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name: "sequential -8 to 7",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = float32(i - 8)
				}
				return v
			}(),
			expected: 7,
		},
		{
			name: "all same",
			input: func() []float32 {
				v := make([]float32, 16)
				for i := range v {
					v[i] = -5.5
				}
				return v
			}(),
			expected: -5.5,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: float32(math.NaN()),
		},
		{
			name:     "positive Inf is max",
			input:    []float32{1, 2, float32(math.Inf(1)), 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: float32(math.Inf(1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x16[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxFloat32x16[float32]()(vec))

			is.NoError(err)
			if math.IsNaN(float64(tc.expected)) {
				is.True(math.IsNaN(float64(result[0])))
			} else if math.IsInf(float64(tc.expected), 1) {
				is.True(math.IsInf(float64(result[0]), 1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

// ==================== Int16x32 tests ====================

func TestAddInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		addend   int16
		expected []int16
	}{
		{
			name: "basic addition",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*100 - 1000)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*100 - 1000)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 30000
				}
				return v
			}(),
			addend: 10000,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -25536 // 40000 - 65536
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x32ToScalar[int16]()(AddInt16x32[int16](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []int16
		subtrahend int16
		expected   []int16
	}{
		{
			name: "basic subtraction",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
			subtrahend: 100,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 400
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*50 - 500)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*50 - 500)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -20000
				}
				return v
			}(),
			subtrahend: 20000,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 25536 // -40000 as int16 wraps to 25536
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x32ToScalar[int16]()(SubInt16x32[int16](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int16
		min       int16
		max       int16
		expected  []int16
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*100 - 1500)
				}
				return v
			}(),
			min: -1000,
			max: 1000,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := 0; i < 32; i++ {
					val := i*100 - 1500
					if val < -1000 {
						v[i] = -1000
					} else if val > 1000 {
						v[i] = 1000
					} else {
						v[i] = int16(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -5000
				}
				return v
			}(),
			min: -1000,
			max: 1000,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -1000
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			min: -1000,
			max: 1000,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []int16{},
			min:       10000,
			max:       5000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampInt16x32[int16](tc.min, tc.max)(ro.Empty[*archsimd.Int16x32]())
				})
				return
			}

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x32ToScalar[int16]()(ClampInt16x32[int16](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int16
		threshold int16
		expected  []int16
	}{
		{
			name: "basic min",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*100 - 1500)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := 0; i < 32; i++ {
					val := i*100 - 1500
					if val < 0 {
						v[i] = 0
					} else {
						v[i] = int16(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x32ToScalar[int16]()(MinInt16x32[int16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int16
		threshold int16
		expected  []int16
	}{
		{
			name: "basic max",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*100 - 1500)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := 0; i < 32; i++ {
					val := i*100 - 1500
					if val > 0 {
						v[i] = 0
					} else {
						v[i] = int16(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x32ToScalar[int16]()(MaxInt16x32[int16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceSumInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name: "sequential 1-32",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i + 1)
				}
				return v
			}(),
			expected: 528, // sum of 1-32
		},
		{
			name:     "all zeros",
			input:    make([]int16, 32),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 32,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt16x32[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestReduceMinInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name: "sequential i*2-30",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*2 - 30)
				}
				return v
			}(),
			expected: -30,
		},
		{
			name: "all same",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			expected: 5000,
		},
		{
			name: "min at end",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = 1000
				}
				v[31] = -32768
				return v
			}(),
			expected: -32768,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt16x32[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name: "sequential i*2-30",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = int16(i*2 - 30)
				}
				return v
			}(),
			expected: 32,
		},
		{
			name: "all same",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
			expected: -500,
		},
		{
			name: "max at start",
			input: func() []int16 {
				v := make([]int16, 32)
				for i := range v {
					v[i] = -1000
				}
				v[0] = 32767
				return v
			}(),
			expected: 32767,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x32[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt16x32[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

// ==================== Int32x16 tests ====================

func TestReduceSumInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential 1-16 * 1000",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i+1) * 1000
				}
				return v
			}(),
			expected: 136000,
		},
		{
			name:     "all zeros",
			input:    make([]int32, 16),
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt32x16[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

// ==================== Int64x8 tests ====================

func TestReduceSumInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "sequential values",
			input:    []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			expected: 36000000,
		},
		{
			name:     "all zeros",
			input:    []int64{0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt64x8[int64]()(vec))

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

// ==================== Uint8x64 ReduceSum tests ====================

func TestReduceSumUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential 1-64",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i + 1)
				}
				return v
			}(),
			expected: 32, // sum of 1-64 = 2080, 2080 % 256 = 32
		},
		{
			name:     "all zeros",
			input:    make([]uint8, 64),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 64,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint8x64[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

// ==================== Uint16x32 tests ====================

func TestReduceSumUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name: "sequential 1-32",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i + 1)
				}
				return v
			}(),
			expected: 528,
		},
		{
			name:     "all zeros",
			input:    make([]uint16, 32),
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint16x32[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

// ==================== Uint32x16 tests ====================

func TestReduceSumUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential 1-16 * 1000",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i+1) * 1000
				}
				return v
			}(),
			expected: 136000,
		},
		{
			name:     "all zeros",
			input:    make([]uint32, 16),
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint32x16[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

// ==================== Uint64x8 tests ====================

func TestReduceSumUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "sequential values",
			input:    []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			expected: 36000000,
		},
		{
			name:     "all zeros",
			input:    []uint64{0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint64x8[uint64]()(vec))

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

// ==================== Float64x8 tests ====================

func TestReduceSumFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "sequential values",
			input:    []float64{1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5},
			expected: 40,
		},
		{
			name:     "all zeros",
			input:    []float64{0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "negative values",
			input:    []float64{-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0},
			expected: -36,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, 2, math.NaN(), 4, 5, 6, 7, 8},
			expected: math.NaN(),
		},
		{
			name:     "positive Inf",
			input:    []float64{1, 2, math.Inf(1), 4, 5, 6, 7, 8},
			expected: math.Inf(1),
		},
		{
			name:     "negative Inf",
			input:    []float64{math.Inf(-1), 2, 3, 4, 5, 6, 7, 8},
			expected: math.Inf(-1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumFloat64x8[float64]()(vec))

			is.NoError(err)
			if math.IsNaN(tc.expected) {
				is.True(math.IsNaN(result[0]))
			} else if math.IsInf(tc.expected, 0) {
				is.Equal(math.IsInf(tc.expected, 1), math.IsInf(result[0], 1))
				is.Equal(math.IsInf(tc.expected, -1), math.IsInf(result[0], -1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

func TestAddFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		addend   float64
		expected []float64
	}{
		{
			name:     "basic addition",
			input:    []float64{1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5},
			addend:   2.5,
			expected: []float64{4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0},
		},
		{
			name:     "add zero",
			input:    []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			addend:   0,
			expected: []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		},
		{
			name:     "negative addend",
			input:    []float64{10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0},
			addend:   -5.0,
			expected: []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x8ToScalar[float64]()(AddFloat64x8[float64](tc.addend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []float64
		subtrahend float64
		expected   []float64
	}{
		{
			name:       "basic subtraction",
			input:      []float64{10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5, 10.5},
			subtrahend: 3.5,
			expected:   []float64{7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0},
		},
		{
			name:       "subtract zero",
			input:      []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
			subtrahend: 0,
			expected:   []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		},
		{
			name:       "negative result",
			input:      []float64{5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0},
			subtrahend: 10.0,
			expected:   []float64{-5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x8ToScalar[float64]()(SubFloat64x8[float64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float64
		min       float64
		max       float64
		expected  []float64
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []float64{-5.5, 0.0, 5.5, 10.5, -10.5, 15.5, -2.5, 7.5},
			min:      -2.5,
			max:      12.5,
			expected: []float64{-2.5, 0.0, 5.5, 10.5, -2.5, 12.5, -2.5, 7.5},
		},
		{
			name:     "all below min",
			input:    []float64{-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0},
			min:      -2.5,
			max:      12.5,
			expected: []float64{-2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5, -2.5},
		},
		{
			name:     "all above max",
			input:    []float64{100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0},
			min:      -2.5,
			max:      12.5,
			expected: []float64{12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5},
		},
		{
			name:      "panic on min > max",
			input:     []float64{},
			min:       10.0,
			max:       5.0,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampFloat64x8[float64](tc.min, tc.max)(ro.Empty[*archsimd.Float64x8]())
				})
				return
			}

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x8ToScalar[float64]()(ClampFloat64x8[float64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic min",
			input:     []float64{-5.5, 0.0, 5.5, 10.5, -10.5, 15.5, -2.5, 7.5},
			threshold: 2.5,
			expected:  []float64{2.5, 2.5, 5.5, 10.5, 2.5, 15.5, 2.5, 7.5},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0},
			threshold: 0,
			expected:  []float64{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "all above threshold",
			input:     []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0},
			threshold: 0,
			expected:  []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0},
		},
		{
			name:      "negative threshold",
			input:     []float64{-5.0, -3.0, 0.0, 3.0, 5.0, -10.0, 10.0, 0.0},
			threshold: -2.0,
			expected:  []float64{-2.0, -2.0, 0.0, 3.0, 5.0, -2.0, 10.0, 0.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x8ToScalar[float64]()(MinFloat64x8[float64](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic max",
			input:     []float64{-5.5, 0.0, 5.5, 10.5, -10.5, 15.5, -2.5, 7.5},
			threshold: 5.5,
			expected:  []float64{-5.5, 0.0, 5.5, 5.5, -10.5, 5.5, -2.5, 5.5},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0},
			threshold: 0,
			expected:  []float64{-10.0, -20.0, -30.0, -40.0, -50.0, -60.0, -70.0, -80.0},
		},
		{
			name:      "all above threshold",
			input:     []float64{10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0},
			threshold: 0,
			expected:  []float64{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "positive threshold",
			input:     []float64{-5.0, -3.0, 0.0, 3.0, 5.0, -10.0, 10.0, 0.0},
			threshold: 2.0,
			expected:  []float64{2.0, 2.0, 2.0, 3.0, 5.0, 2.0, 2.0, 2.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x8ToScalar[float64]()(MaxFloat64x8[float64](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceMinFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "mixed values",
			input:    []float64{3.5, 1.5, 4.5, 2.5, 5.5, 0.5, 3.5, 2.5},
			expected: 0.5,
		},
		{
			name: "all positive",
			input: func() []float64 {
				v := make([]float64, 8)
				for i := range v {
					v[i] = float64(i + 1)
				}
				return v
			}(),
			expected: 1.0,
		},
		{
			name:     "all negative",
			input:    []float64{-10.0, -5.0, -20.0, -15.0, -1.0, -100.0, -50.0, -25.0},
			expected: -100.0,
		},
		{
			name:     "single min at end",
			input:    []float64{100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5, 0.1},
			expected: 0.1,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, 2, math.NaN(), 4, 5, 6, 7, 8},
			expected: math.NaN(),
		},
		{
			name:     "negative Inf is min",
			input:    []float64{1, 2, math.Inf(-1), 4, 5, 6, 7, 8},
			expected: math.Inf(-1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinFloat64x8[float64]()(vec))

			is.NoError(err)
			is.Len(result, 1)
			if math.IsNaN(tc.expected) {
				is.True(math.IsNaN(result[0]))
			} else if math.IsInf(tc.expected, -1) {
				is.True(math.IsInf(result[0], -1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

func TestReduceMaxFloat64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "mixed values",
			input:    []float64{3.5, 1.5, 4.5, 2.5, 5.5, 0.5, 3.5, 2.5},
			expected: 5.5,
		},
		{
			name: "sequential",
			input: func() []float64 {
				v := make([]float64, 8)
				for i := range v {
					v[i] = float64(i + 1)
				}
				return v
			}(),
			expected: 8.0,
		},
		{
			name:     "all negative",
			input:    []float64{-10.0, -5.0, -20.0, -15.0, -1.0, -100.0, -50.0, -25.0},
			expected: -1.0,
		},
		{
			name:     "single max at start",
			input:    []float64{100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5, 0.1},
			expected: 100.0,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, 2, math.NaN(), 4, 5, 6, 7, 8},
			expected: math.NaN(),
		},
		{
			name:     "positive Inf is max",
			input:    []float64{1, 2, math.Inf(1), 4, 5, 6, 7, 8},
			expected: math.Inf(1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x8[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxFloat64x8[float64]()(vec))

			is.NoError(err)
			is.Len(result, 1)
			if math.IsNaN(tc.expected) {
				is.True(math.IsNaN(result[0]))
			} else if math.IsInf(tc.expected, 1) {
				is.True(math.IsInf(result[0], 1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

// ==================== Int32x16 tests ====================

func TestAddInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		addend   int32
		expected []int32
	}{
		{
			name: "basic addition",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			addend: 5,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 15
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = -1000
					} else {
						v[i] = 1000
					}
				}
				return v
			}(),
			addend: 500,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = -500
					} else {
						v[i] = 1500
					}
				}
				return v
			}(),
		},
		{
			name: "boundary values",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = 2147483647
					} else {
						v[i] = -2147483648
					}
				}
				return v
			}(),
			addend: 1,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = -2147483648
					} else {
						v[i] = -2147483647
					}
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x16ToScalar[int32]()(AddInt32x16[int32](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []int32
		subtrahend int32
		expected   []int32
	}{
		{
			name: "basic subtraction",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			subtrahend: 10,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 40
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			subtrahend: 200,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
		},
		{
			name: "boundary values",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = -2147483648
					} else {
						v[i] = 2147483647
					}
				}
				return v
			}(),
			subtrahend: 1,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					if i%2 == 0 {
						v[i] = 2147483647
					} else {
						v[i] = 2147483646
					}
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x16ToScalar[int32]()(SubInt32x16[int32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int32
		min       int32
		max       int32
		expected  []int32
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			min: -5,
			max: 5,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := 0; i < 16; i++ {
					val := i - 8
					if val < -5 {
						v[i] = -5
					} else if val > 5 {
						v[i] = 5
					} else {
						v[i] = int32(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -10000
				}
				return v
			}(),
			min: -5000,
			max: 5000,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -5000
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 10000
				}
				return v
			}(),
			min: -5000,
			max: 5000,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []int32{},
			min:       10000,
			max:       5000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampInt32x16[int32](tc.min, tc.max)(ro.Empty[*archsimd.Int32x16]())
				})
				return
			}

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x16ToScalar[int32]()(ClampInt32x16[int32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int32
		threshold int32
		expected  []int32
	}{
		{
			name: "basic min",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := 0; i < 16; i++ {
					val := i - 8
					if val < 0 {
						v[i] = 0
					} else {
						v[i] = int32(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -1000
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x16ToScalar[int32]()(MinInt32x16[int32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int32
		threshold int32
		expected  []int32
	}{
		{
			name: "basic max",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := 0; i < 16; i++ {
					val := i - 8
					if val > 0 {
						v[i] = 0
					} else {
						v[i] = int32(val)
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -1000
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -1000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x16ToScalar[int32]()(MaxInt32x16[int32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential -8 to 7",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			expected: -8,
		},
		{
			name: "all same",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			expected: 5000,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt32x16[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential -8 to 7",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = int32(i - 8)
				}
				return v
			}(),
			expected: 7,
		},
		{
			name: "all same",
			input: func() []int32 {
				v := make([]int32, 16)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
			expected: -500,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x16[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt32x16[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

// ==================== Int64x8 tests ====================

func TestAddInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		addend   int64
		expected []int64
	}{
		{
			name:     "basic addition",
			input:    []int64{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000},
			addend:   500000,
			expected: []int64{1500000, 1500000, 1500000, 1500000, 1500000, 1500000, 1500000, 1500000},
		},
		{
			name:     "add zero",
			input:    []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			addend:   0,
			expected: []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x8ToScalar[int64]()(AddInt64x8[int64](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []int64
		subtrahend int64
		expected   []int64
	}{
		{
			name:       "basic subtraction",
			input:      []int64{5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000},
			subtrahend: 2000000,
			expected:   []int64{3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000},
		},
		{
			name:       "subtract zero",
			input:      []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			subtrahend: 0,
			expected:   []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x8ToScalar[int64]()(SubInt64x8[int64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	baseInput := []int64{-2000000, -1000000, 1000000, 2000000, -3000000, 3000000, -1500000, 1500000}
	testCases := []struct {
		name      string
		input     []int64
		min       int64
		max       int64
		expected  []int64
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    baseInput,
			min:      -1500000,
			max:      1500000,
			expected: []int64{-1500000, -1000000, 1000000, 1500000, -1500000, 1500000, -1500000, 1500000},
		},
		{
			name:      "panic on min > max",
			input:     []int64{},
			min:       10000000,
			max:       5000000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampInt64x8[int64](tc.min, tc.max)(ro.Empty[*archsimd.Int64x8]())
				})
				return
			}

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x8ToScalar[int64]()(ClampInt64x8[int64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int64
		threshold int64
		expected  []int64
	}{
		{
			name:      "basic min",
			input:     []int64{-2000000, -1000000, 1000000, 2000000, -3000000, 3000000, -1500000, 1500000},
			threshold: 0,
			expected:  []int64{0, 0, 1000000, 2000000, 0, 3000000, 0, 1500000},
		},
		{
			name:      "all above threshold",
			input:     []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			threshold: 0,
			expected:  []int64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x8ToScalar[int64]()(MinInt64x8[int64](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int64
		threshold int64
		expected  []int64
	}{
		{
			name:      "basic max",
			input:     []int64{-2000000, -1000000, 1000000, 2000000, -3000000, 3000000, -1500000, 1500000},
			threshold: 0,
			expected:  []int64{-2000000, -1000000, 0, 0, -3000000, 0, -1500000, 0},
		},
		{
			name:      "all below threshold",
			input:     []int64{-1000000, -2000000, -3000000, -4000000, -5000000, -6000000, -7000000, -8000000},
			threshold: 0,
			expected:  []int64{-1000000, -2000000, -3000000, -4000000, -5000000, -6000000, -7000000, -8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x8ToScalar[int64]()(MaxInt64x8[int64](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "mixed values",
			input:    []int64{-2000000, -1000000, 1000000, 2000000, -3000000, 3000000, -1500000, 1500000},
			expected: -3000000,
		},
		{
			name:     "all same",
			input:    []int64{5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000},
			expected: 5000000,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt64x8[int64]()(vec))

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "mixed values",
			input:    []int64{-2000000, -1000000, 1000000, 2000000, -3000000, 3000000, -1500000, 1500000},
			expected: 3000000,
		},
		{
			name:     "all same",
			input:    []int64{-500, -500, -500, -500, -500, -500, -500, -500},
			expected: -500,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x8[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt64x8[int64]()(vec))

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

// ==================== Uint8x64 tests ====================

func TestAddUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		addend   uint8
		expected []uint8
	}{
		{
			name: "basic addition",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			addend: 5,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 15
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 250
				}
				return v
			}(),
			addend: 20,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 14 // 250+20 = 270 % 256 = 14
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x64ToScalar[uint8]()(AddUint8x64[uint8](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []uint8
		subtrahend uint8
		expected   []uint8
	}{
		{
			name: "basic subtraction",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			subtrahend: 10,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 40
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			subtrahend: 20,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 241 // 5-20 as uint8 = 256-15 = 241
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x64ToScalar[uint8]()(SubUint8x64[uint8](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint8
		min       uint8
		max       uint8
		expected  []uint8
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			min: 10,
			max: 50,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := 0; i < 64; i++ {
					val := uint8(i)
					if val < 10 {
						v[i] = 10
					} else if val > 50 {
						v[i] = 50
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			min: 10,
			max: 50,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
			min: 10,
			max: 50,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []uint8{},
			min:       100,
			max:       50,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampUint8x64[uint8](tc.min, tc.max)(ro.Empty[*archsimd.Uint8x64]())
				})
				return
			}

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x64ToScalar[uint8]()(ClampUint8x64[uint8](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name: "basic min",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := 0; i < 64; i++ {
					val := uint8(i)
					if val < 30 {
						v[i] = 30
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 30
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x64ToScalar[uint8]()(MinUint8x64[uint8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name: "basic max",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := 0; i < 64; i++ {
					val := uint8(i)
					if val > 30 {
						v[i] = 30
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 30,
			expected: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 30
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x64ToScalar[uint8]()(MaxUint8x64[uint8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential 10-73",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i + 10)
				}
				return v
			}(),
			expected: 10,
		},
		{
			name: "all same",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			expected: 100,
		},
		{
			name: "min at end",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 200
				}
				v[63] = 0
				return v
			}(),
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint8x64[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint8x64(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential 10-73",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = uint8(i + 10)
				}
				return v
			}(),
			expected: 73,
		},
		{
			name: "all same",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			expected: 50,
		},
		{
			name: "max at start",
			input: func() []uint8 {
				v := make([]uint8, 64)
				for i := range v {
					v[i] = 1
				}
				v[0] = 255
				return v
			}(),
			expected: 255,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x64[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint8x64[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

// ==================== Uint16x32 tests ====================

func TestAddUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		addend   uint16
		expected []uint16
	}{
		{
			name: "basic addition",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			addend: 20000,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 4528 // 70000 % 65536 = 4528
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x32ToScalar[uint16]()(AddUint16x32[uint16](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []uint16
		subtrahend uint16
		expected   []uint16
	}{
		{
			name: "basic subtraction",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
			subtrahend: 100,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 400
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			subtrahend: 500,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 65136 // -400 as uint16
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x32ToScalar[uint16]()(SubUint16x32[uint16](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint16
		min       uint16
		max       uint16
		expected  []uint16
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := 0; i < 32; i++ {
					val := uint16(i * 10)
					if val < 50 {
						v[i] = 50
					} else if val > 200 {
						v[i] = 200
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []uint16{},
			min:       10000,
			max:       5000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampUint16x32[uint16](tc.min, tc.max)(ro.Empty[*archsimd.Uint16x32]())
				})
				return
			}

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x32ToScalar[uint16]()(ClampUint16x32[uint16](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint16
		threshold uint16
		expected  []uint16
	}{
		{
			name: "basic min",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := 0; i < 32; i++ {
					val := uint16(i * 10)
					if val < 150 {
						v[i] = 150
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x32ToScalar[uint16]()(MinUint16x32[uint16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint16
		threshold uint16
		expected  []uint16
	}{
		{
			name: "basic max",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i * 10)
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := 0; i < 32; i++ {
					val := uint16(i * 10)
					if val > 150 {
						v[i] = 150
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
			threshold: 150,
			expected: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x32ToScalar[uint16]()(MaxUint16x32[uint16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name: "sequential 100-131",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i + 100)
				}
				return v
			}(),
			expected: 100,
		},
		{
			name: "all same",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			expected: 5000,
		},
		{
			name: "min at end",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 1000
				}
				v[31] = 0
				return v
			}(),
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint16x32[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint16x32(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name: "sequential 100-131",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = uint16(i + 100)
				}
				return v
			}(),
			expected: 131,
		},
		{
			name: "all same",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			expected: 100,
		},
		{
			name: "max at start",
			input: func() []uint16 {
				v := make([]uint16, 32)
				for i := range v {
					v[i] = 1
				}
				v[0] = 65535
				return v
			}(),
			expected: 65535,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x32[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint16x32[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

// ==================== Uint32x16 tests ====================

func TestAddUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		addend   uint32
		expected []uint32
	}{
		{
			name: "basic addition",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			addend: 500,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1500
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 4000000000
				}
				return v
			}(),
			addend: 500000000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 205032704 // 4.5e9 % 2^32
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x16ToScalar[uint32]()(AddUint32x16[uint32](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []uint32
		subtrahend uint32
		expected   []uint32
	}{
		{
			name: "basic subtraction",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			subtrahend: 1000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 4000
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			subtrahend: 500,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 4294966896 // -400 as uint32
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x16ToScalar[uint32]()(SubUint32x16[uint32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint32
		min       uint32
		max       uint32
		expected  []uint32
		wantPanic bool
	}{
		{
			name: "basic clamp",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := 0; i < 16; i++ {
					val := uint32(i * 1000)
					if val < 5000 {
						v[i] = 5000
					} else if val > 10000 {
						v[i] = 10000
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below min",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 10000
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []uint32{},
			min:       10000,
			max:       5000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampUint32x16[uint32](tc.min, tc.max)(ro.Empty[*archsimd.Uint32x16]())
				})
				return
			}

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x16ToScalar[uint32]()(ClampUint32x16[uint32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint32
		threshold uint32
		expected  []uint32
	}{
		{
			name: "basic min",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := 0; i < 16; i++ {
					val := uint32(i * 1000)
					if val < 7000 {
						v[i] = 7000
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 7000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 20000
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 20000
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x16ToScalar[uint32]()(MinUint32x16[uint32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint32
		threshold uint32
		expected  []uint32
	}{
		{
			name: "basic max",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := 0; i < 16; i++ {
					val := uint32(i * 1000)
					if val > 7000 {
						v[i] = 7000
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "all below threshold",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 20000
				}
				return v
			}(),
			threshold: 7000,
			expected: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 7000
				}
				return v
			}(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x16ToScalar[uint32]()(MaxUint32x16[uint32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential 10000-10015",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i + 10000)
				}
				return v
			}(),
			expected: 10000,
		},
		{
			name: "all same",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			expected: 50000,
		},
		{
			name: "min at end",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 100000
				}
				v[15] = 1
				return v
			}(),
			expected: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint32x16[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint32x16(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential 10000-10015",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = uint32(i + 10000)
				}
				return v
			}(),
			expected: 10015,
		},
		{
			name: "all same",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			expected: 100,
		},
		{
			name: "max at start",
			input: func() []uint32 {
				v := make([]uint32, 16)
				for i := range v {
					v[i] = 1
				}
				v[0] = 4294967295
				return v
			}(),
			expected: 4294967295,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x16[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint32x16[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

// ==================== Uint64x8 tests ====================

func TestAddUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		addend   uint64
		expected []uint64
	}{
		{
			name:     "basic addition",
			input:    []uint64{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000},
			addend:   500000,
			expected: []uint64{1500000, 1500000, 1500000, 1500000, 1500000, 1500000, 1500000, 1500000},
		},
		{
			name:     "add zero",
			input:    []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			addend:   0,
			expected: []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x8ToScalar[uint64]()(AddUint64x8[uint64](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name       string
		input      []uint64
		subtrahend uint64
		expected   []uint64
	}{
		{
			name:       "basic subtraction",
			input:      []uint64{5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000},
			subtrahend: 2000000,
			expected:   []uint64{3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000, 3000000},
		},
		{
			name:       "subtract zero",
			input:      []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			subtrahend: 0,
			expected:   []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x8ToScalar[uint64]()(SubUint64x8[uint64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		min       uint64
		max       uint64
		expected  []uint64
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			min:      2500000,
			max:      6500000,
			expected: []uint64{2500000, 2500000, 3000000, 4000000, 5000000, 6000000, 6500000, 6500000},
		},
		{
			name:     "all below min",
			input:    []uint64{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000},
			min:      2500000,
			max:      6500000,
			expected: []uint64{2500000, 2500000, 2500000, 2500000, 2500000, 2500000, 2500000, 2500000},
		},
		{
			name:     "all above max",
			input:    []uint64{8000000, 8000000, 8000000, 8000000, 8000000, 8000000, 8000000, 8000000},
			min:      2500000,
			max:      6500000,
			expected: []uint64{6500000, 6500000, 6500000, 6500000, 6500000, 6500000, 6500000, 6500000},
		},
		{
			name:      "panic on min > max",
			input:     []uint64{},
			min:       10000000,
			max:       5000000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampUint64x8[uint64](tc.min, tc.max)(ro.Empty[*archsimd.Uint64x8]())
				})
				return
			}

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x8ToScalar[uint64]()(ClampUint64x8[uint64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		threshold uint64
		expected  []uint64
	}{
		{
			name:      "basic min",
			input:     []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			threshold: 4500000,
			expected:  []uint64{4500000, 4500000, 4500000, 4500000, 5000000, 6000000, 7000000, 8000000},
		},
		{
			name:      "all below threshold",
			input:     []uint64{1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000, 1000000},
			threshold: 4500000,
			expected:  []uint64{4500000, 4500000, 4500000, 4500000, 4500000, 4500000, 4500000, 4500000},
		},
		{
			name:      "all above threshold",
			input:     []uint64{5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000},
			threshold: 4500000,
			expected:  []uint64{5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x8ToScalar[uint64]()(MinUint64x8[uint64](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		threshold uint64
		expected  []uint64
	}{
		{
			name:      "basic max",
			input:     []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			threshold: 4500000,
			expected:  []uint64{1000000, 2000000, 3000000, 4000000, 4500000, 4500000, 4500000, 4500000},
		},
		{
			name:      "all below threshold",
			input:     []uint64{1000000, 2000000, 3000000, 4000000, 1000000, 2000000, 3000000, 4000000},
			threshold: 4500000,
			expected:  []uint64{1000000, 2000000, 3000000, 4000000, 1000000, 2000000, 3000000, 4000000},
		},
		{
			name:      "all above threshold",
			input:     []uint64{5000000, 6000000, 7000000, 8000000, 9000000, 10000000, 11000000, 12000000},
			threshold: 4500000,
			expected:  []uint64{4500000, 4500000, 4500000, 4500000, 4500000, 4500000, 4500000, 4500000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x8ToScalar[uint64]()(MaxUint64x8[uint64](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "sequential values",
			input:    []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			expected: 1000000,
		},
		{
			name:     "all same",
			input:    []uint64{5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000},
			expected: 5000000,
		},
		{
			name:     "min at end",
			input:    []uint64{8000000, 7000000, 6000000, 5000000, 4000000, 3000000, 2000000, 1},
			expected: 1,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint64x8[uint64]()(vec))

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint64x8(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "sequential values",
			input:    []uint64{1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000, 8000000},
			expected: 8000000,
		},
		{
			name:     "all same",
			input:    []uint64{5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000, 5000000},
			expected: 5000000,
		},
		{
			name:     "max at start",
			input:    []uint64{18446744073709551615, 1, 1, 1, 1, 1, 1, 1},
			expected: 18446744073709551615,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x8[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint64x8[uint64]()(vec))

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

// ==================== Int64x4 Min/Max/ReduceMin/ReduceMax tests ====================
// NOTE: These operations require AVX-512 because AVX2 doesn't have 64-bit integer comparison instructions

func TestClampInt64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []int64
		min      int64
		max      int64
		expected []int64
	}{
		{
			name:     "basic clamping",
			input:    []int64{-2000000, -1000000, 1000000, 2000000},
			min:      -1500000,
			max:      1500000,
			expected: []int64{-1500000, -1000000, 1000000, 1500000},
		},
		{
			name:     "all below min",
			input:    []int64{-5000000, -4000000, -3000000, -2000000},
			min:      -1500000,
			max:      1500000,
			expected: []int64{-1500000, -1500000, -1500000, -1500000},
		},
		{
			name:     "all above max",
			input:    []int64{3000000, 4000000, 5000000, 6000000},
			min:      -1500000,
			max:      1500000,
			expected: []int64{1500000, 1500000, 1500000, 1500000},
		},
		{
			name:     "all in range",
			input:    []int64{-100000, 0, 100000, 500000},
			min:      -1500000,
			max:      1500000,
			expected: []int64{-100000, 0, 100000, 500000},
		},
		{
			name:     "zero bounds",
			input:    []int64{-1000000, -500000, 500000, 1000000},
			min:      0,
			max:      0,
			expected: []int64{0, 0, 0, 0},
		},
		{
			name:     "negative range",
			input:    []int64{-5000000, -3000000, -2000000, -1000000},
			min:      -4000000,
			max:      -2000000,
			expected: []int64{-4000000, -3000000, -2000000, -2000000},
		},
		{
			name:     "positive range",
			input:    []int64{0, 1000000, 2000000, 3000000},
			min:      500000,
			max:      2500000,
			expected: []int64{500000, 1000000, 2000000, 2500000},
		},
		{
			name:     "large values",
			input:    []int64{-9223372036854775808, 0, 5000000000000000, 9223372036854775807},
			min:      -10000000000000000,
			max:      10000000000000000,
			expected: []int64{-10000000000000000, 0, 5000000000000000, 10000000000000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(ClampInt64x4[int64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}

	// Test panic behavior
	t.Run("panic on min > max", func(t *testing.T) {
		is := assert.New(t)
		is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
			_ = ClampInt64x4[int64](10000000, 5000000)(ro.Empty[*archsimd.Int64x4]())
		})
	})
}

func TestMinInt64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []int64
		minValue int64
		expected []int64
	}{
		{
			name:     "basic min with zero",
			input:    []int64{-2000000, -1000000, 1000000, 2000000},
			minValue: 0,
			expected: []int64{0, 0, 1000000, 2000000},
		},
		{
			name:     "all below min",
			input:    []int64{-5000000, -4000000, -3000000, -2000000},
			minValue: -1000000,
			expected: []int64{-1000000, -1000000, -1000000, -1000000},
		},
		{
			name:     "all above min",
			input:    []int64{3000000, 4000000, 5000000, 6000000},
			minValue: 1000000,
			expected: []int64{3000000, 4000000, 5000000, 6000000},
		},
		{
			name:     "negative min",
			input:    []int64{-3000000, -2000000, -1000000, 0},
			minValue: -2500000,
			expected: []int64{-2500000, -2000000, -1000000, 0},
		},
		{
			name:     "positive min",
			input:    []int64{0, 1000000, 2000000, 3000000},
			minValue: 1500000,
			expected: []int64{1500000, 1500000, 2000000, 3000000},
		},
		{
			name:     "large values",
			input:    []int64{-9223372036854775808, 0, 5000000000000000, 9223372036854775807},
			minValue: 1000000000000000,
			expected: []int64{1000000000000000, 1000000000000000, 5000000000000000, 9223372036854775807},
		},
		{
			name:     "min at int64 max",
			input:    []int64{0, 1000000, 2000000, 3000000},
			minValue: 9223372036854775807,
			expected: []int64{9223372036854775807, 9223372036854775807, 9223372036854775807, 9223372036854775807},
		},
		{
			name:     "min at int64 min",
			input:    []int64{-1000000, 0, 1000000, 2000000},
			minValue: -9223372036854775808,
			expected: []int64{-1000000, 0, 1000000, 2000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(MinInt64x4[int64](tc.minValue)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []int64
		maxValue int64
		expected []int64
	}{
		{
			name:     "basic max with zero",
			input:    []int64{-2000000, -1000000, 1000000, 2000000},
			maxValue: 0,
			expected: []int64{-2000000, -1000000, 0, 0},
		},
		{
			name:     "all below max",
			input:    []int64{-5000000, -4000000, -3000000, -2000000},
			maxValue: -1000000,
			expected: []int64{-5000000, -4000000, -3000000, -2000000},
		},
		{
			name:     "all above max",
			input:    []int64{3000000, 4000000, 5000000, 6000000},
			maxValue: 3500000,
			expected: []int64{3000000, 3500000, 3500000, 3500000},
		},
		{
			name:     "negative max",
			input:    []int64{-5000000, -3000000, -2000000, -1000000},
			maxValue: -2500000,
			expected: []int64{-5000000, -3000000, -2500000, -2500000},
		},
		{
			name:     "positive max",
			input:    []int64{0, 1000000, 2000000, 3000000},
			maxValue: 1500000,
			expected: []int64{0, 1000000, 1500000, 1500000},
		},
		{
			name:     "large values",
			input:    []int64{-9223372036854775808, 0, 5000000000000000, 9223372036854775807},
			maxValue: 1000000000000000,
			expected: []int64{-9223372036854775808, 0, 1000000000000000, 1000000000000000},
		},
		{
			name:     "max at int64 max",
			input:    []int64{0, 1000000, 2000000, 3000000},
			maxValue: 9223372036854775807,
			expected: []int64{0, 1000000, 2000000, 3000000},
		},
		{
			name:     "max at int64 min",
			input:    []int64{-1000000, 0, 1000000, 2000000},
			maxValue: -9223372036854775808,
			expected: []int64{-9223372036854775808, -9223372036854775808, -9223372036854775808, -9223372036854775808},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(MaxInt64x4[int64](tc.maxValue)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "basic mixed values",
			input:    []int64{-2000000, -1000000, 1000000, 2000000},
			expected: -2000000,
		},
		{
			name:     "all positive values - detects accumulator initialization bug",
			input:    []int64{1, 2, 3, 4},
			expected: 1,
		},
		{
			name:     "empty observable returns 0",
			input:    []int64{},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int64
			var err error

			if len(tc.input) == 0 {
				result, err = ro.Collect(
					ReduceMinInt64x4[int64]()(ro.Empty[*archsimd.Int64x4]()),
				)
			} else {
				vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
				result, err = ro.Collect(ReduceMinInt64x4[int64]()(vec))
			}

			is.NoError(err)
			if len(tc.input) == 0 {
				is.Equal([]int64{tc.expected}, result)
			} else {
				is.Equal([]int64{tc.expected}, result)
			}
		})
	}
}

func TestReduceMaxInt64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "basic mixed values",
			input:    []int64{-2000000, -1000000, 1000000, 2000000},
			expected: 2000000,
		},
		{
			name:     "all negative values - detects accumulator initialization bug",
			input:    []int64{-1, -2, -3, -4},
			expected: -1,
		},
		{
			name:     "empty observable returns 0",
			input:    []int64{},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int64
			var err error

			if len(tc.input) == 0 {
				result, err = ro.Collect(
					ReduceMaxInt64x4[int64]()(ro.Empty[*archsimd.Int64x4]()),
				)
			} else {
				vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
				result, err = ro.Collect(ReduceMaxInt64x4[int64]()(vec))
			}

			is.NoError(err)
			if len(tc.input) == 0 {
				is.Equal([]int64{tc.expected}, result)
			} else {
				is.Equal([]int64{tc.expected}, result)
			}
		})
	}
}

// ==================== Uint64x4 Min/Max/ReduceMin/ReduceMax tests ====================
// NOTE: These operations require AVX-512 because AVX2 doesn't have 64-bit integer comparison instructions

func TestClampUint64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []uint64
		min      uint64
		max      uint64
		expected []uint64
	}{
		{
			name:     "basic clamping",
			input:    []uint64{0, 20000000, 40000000, 60000000},
			min:      20000000,
			max:      50000000,
			expected: []uint64{20000000, 20000000, 40000000, 50000000},
		},
		{
			name:     "all below min",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			min:      20000000,
			max:      50000000,
			expected: []uint64{20000000, 20000000, 20000000, 20000000},
		},
		{
			name:     "all above max",
			input:    []uint64{60000000, 70000000, 80000000, 90000000},
			min:      20000000,
			max:      50000000,
			expected: []uint64{50000000, 50000000, 50000000, 50000000},
		},
		{
			name:     "all in range",
			input:    []uint64{30000000, 35000000, 40000000, 45000000},
			min:      20000000,
			max:      50000000,
			expected: []uint64{30000000, 35000000, 40000000, 45000000},
		},
		{
			name:     "zero bounds",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			min:      0,
			max:      0,
			expected: []uint64{0, 0, 0, 0},
		},
		{
			name:     "small values",
			input:    []uint64{0, 1, 2, 3},
			min:      1,
			max:      2,
			expected: []uint64{1, 1, 2, 2},
		},
		{
			name:     "large values near max uint64",
			input:    []uint64{18446744073709551613, 18446744073709551614, 18446744073709551615, 18446744073709551615},
			min:      18446744073709551614,
			max:      18446744073709551615,
			expected: []uint64{18446744073709551614, 18446744073709551614, 18446744073709551615, 18446744073709551615},
		},
		{
			name:     "power of 2 values",
			input:    []uint64{0, 1024, 2048, 4096},
			min:      1500,
			max:      3000,
			expected: []uint64{1500, 1500, 2048, 3000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(ClampUint64x4[uint64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}

	// Test panic behavior
	t.Run("panic on min > max", func(t *testing.T) {
		is := assert.New(t)
		is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
			_ = ClampUint64x4[uint64](10000000, 5000000)(ro.Empty[*archsimd.Uint64x4]())
		})
	})
}

func TestMinUint64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []uint64
		minValue uint64
		expected []uint64
	}{
		{
			name:     "basic min",
			input:    []uint64{0, 20000000, 40000000, 60000000},
			minValue: 30000000,
			expected: []uint64{30000000, 30000000, 40000000, 60000000},
		},
		{
			name:     "all below min",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			minValue: 20000000,
			expected: []uint64{20000000, 20000000, 20000000, 20000000},
		},
		{
			name:     "all above min",
			input:    []uint64{30000000, 40000000, 50000000, 60000000},
			minValue: 10000000,
			expected: []uint64{30000000, 40000000, 50000000, 60000000},
		},
		{
			name:     "zero min",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			minValue: 0,
			expected: []uint64{0, 5000000, 10000000, 15000000},
		},
		{
			name:     "small values",
			input:    []uint64{0, 1, 2, 3},
			minValue: 2,
			expected: []uint64{2, 2, 2, 3},
		},
		{
			name:     "large values near max uint64",
			input:    []uint64{18446744073709551613, 18446744073709551614, 18446744073709551615, 18446744073709551615},
			minValue: 18446744073709551614,
			expected: []uint64{18446744073709551614, 18446744073709551614, 18446744073709551615, 18446744073709551615},
		},
		{
			name:     "power of 2 values",
			input:    []uint64{0, 1024, 2048, 4096},
			minValue: 2000,
			expected: []uint64{2000, 2000, 2048, 4096},
		},
		{
			name:     "min at uint64 max",
			input:    []uint64{0, 1000000, 2000000, 3000000},
			minValue: 18446744073709551615,
			expected: []uint64{18446744073709551615, 18446744073709551615, 18446744073709551615, 18446744073709551615},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(MinUint64x4[uint64](tc.minValue)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []uint64
		maxValue uint64
		expected []uint64
	}{
		{
			name:     "basic max",
			input:    []uint64{0, 20000000, 40000000, 60000000},
			maxValue: 30000000,
			expected: []uint64{0, 20000000, 30000000, 30000000},
		},
		{
			name:     "all below max",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			maxValue: 20000000,
			expected: []uint64{0, 5000000, 10000000, 15000000},
		},
		{
			name:     "all above max",
			input:    []uint64{30000000, 40000000, 50000000, 60000000},
			maxValue: 25000000,
			expected: []uint64{25000000, 25000000, 25000000, 25000000},
		},
		{
			name:     "zero max",
			input:    []uint64{0, 5000000, 10000000, 15000000},
			maxValue: 0,
			expected: []uint64{0, 0, 0, 0},
		},
		{
			name:     "small values",
			input:    []uint64{0, 1, 2, 3},
			maxValue: 2,
			expected: []uint64{0, 1, 2, 2},
		},
		{
			name:     "large values near max uint64",
			input:    []uint64{18446744073709551613, 18446744073709551614, 18446744073709551615, 18446744073709551615},
			maxValue: 18446744073709551614,
			expected: []uint64{18446744073709551613, 18446744073709551614, 18446744073709551614, 18446744073709551614},
		},
		{
			name:     "power of 2 values",
			input:    []uint64{0, 1024, 2048, 4096},
			maxValue: 3000,
			expected: []uint64{0, 1024, 2048, 3000},
		},
		{
			name:     "max at uint64 max",
			input:    []uint64{0, 1000000, 2000000, 3000000},
			maxValue: 18446744073709551615,
			expected: []uint64{0, 1000000, 2000000, 3000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(MaxUint64x4[uint64](tc.maxValue)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "basic mixed values",
			input:    []uint64{60000000, 40000000, 20000000, 80000000},
			expected: 20000000,
		},
		{
			name:     "all same value",
			input:    []uint64{12345, 12345, 12345, 12345},
			expected: 12345,
		},
		{
			name:     "empty observable returns 0",
			input:    []uint64{},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []uint64
			var err error

			if len(tc.input) == 0 {
				result, err = ro.Collect(
					ReduceMinUint64x4[uint64]()(ro.Empty[*archsimd.Uint64x4]()),
				)
			} else {
				vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
				result, err = ro.Collect(ReduceMinUint64x4[uint64]()(vec))
			}

			is.NoError(err)
			if len(tc.input) == 0 {
				is.Equal([]uint64{tc.expected}, result)
			} else {
				is.Equal([]uint64{tc.expected}, result)
			}
		})
	}
}

func TestReduceMaxUint64x4(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "basic mixed values",
			input:    []uint64{60000000, 40000000, 20000000, 80000000},
			expected: 80000000,
		},
		{
			name:     "all same value",
			input:    []uint64{12345, 12345, 12345, 12345},
			expected: 12345,
		},
		{
			name:     "empty observable returns 0",
			input:    []uint64{},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []uint64
			var err error

			if len(tc.input) == 0 {
				result, err = ro.Collect(
					ReduceMaxUint64x4[uint64]()(ro.Empty[*archsimd.Uint64x4]()),
				)
			} else {
				vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
				result, err = ro.Collect(ReduceMaxUint64x4[uint64]()(vec))
			}

			is.NoError(err)
			if len(tc.input) == 0 {
				is.Equal([]uint64{tc.expected}, result)
			} else {
				is.Equal([]uint64{tc.expected}, result)
			}
		})
	}
}

// ==================== Int64x2 and Uint64x2 Min/Max/Clamp tests ====================
// NOTE: These operations require AVX-512 because SSE/AVX2 don't have 64-bit integer comparison instructions

func TestClampInt64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int64
		min       int64
		max       int64
		expected  []int64
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []int64{-1500000, 500000},
			min:      -500000,
			max:      1500000,
			expected: []int64{-500000, 500000},
		},
		{
			name:     "all below min",
			input:    []int64{-2000000, -3000000},
			min:      -500000,
			max:      1500000,
			expected: []int64{-500000, -500000},
		},
		{
			name:     "all above max",
			input:    []int64{2000000, 3000000},
			min:      -500000,
			max:      1500000,
			expected: []int64{1500000, 1500000},
		},
		{
			name:      "panic on min > max",
			input:     []int64{},
			min:       10000000,
			max:       5000000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampInt64x2[int64](tc.min, tc.max)(ro.Empty[*archsimd.Int64x2]())
				})
				return
			}

			vec := archsimd.Int64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int64x2ToScalar[int64]()(ClampInt64x2[int64](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int64
		threshold int64
		expected  []int64
	}{
		{
			name:      "basic min",
			input:     []int64{-1500000, 500000},
			threshold: 0,
			expected:  []int64{0, 500000},
		},
		{
			name:      "all below threshold",
			input:     []int64{-2000000, -1000000},
			threshold: 0,
			expected:  []int64{0, 0},
		},
		{
			name:      "all above threshold",
			input:     []int64{1000000, 2000000},
			threshold: 0,
			expected:  []int64{1000000, 2000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int64x2ToScalar[int64]()(MinInt64x2[int64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []int64
		threshold int64
		expected  []int64
	}{
		{
			name:      "basic max",
			input:     []int64{-1500000, 500000},
			threshold: 0,
			expected:  []int64{-1500000, 0},
		},
		{
			name:      "all below threshold",
			input:     []int64{-2000000, -1000000},
			threshold: 0,
			expected:  []int64{-2000000, -1000000},
		},
		{
			name:      "all above threshold",
			input:     []int64{1000000, 2000000},
			threshold: 0,
			expected:  []int64{0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int64x2ToScalar[int64]()(MaxInt64x2[int64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt64x2(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int64x2
		expected int64
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int64x2 {
				vec := archsimd.Int64x2{}
				for i := 0; i < 2; i++ {
					vec = vec.SetElem(uint8(i), int64(int64(i)*1000000-1500000))
				}
				return &vec
			},
			expected: -1500000,
		},
		{
			name: "all positive values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int64x2 {
				vec := archsimd.Int64x2{}
				for i := 0; i < 2; i++ {
					vec = vec.SetElem(uint8(i), int64(i+1))
				}
				return &vec
			},
			expected: 1,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int64x2 {
				return nil
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int64
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMinInt64x2[int64]()(ro.Empty[*archsimd.Int64x2]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMinInt64x2[int64]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt64x2(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int64x2
		expected int64
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int64x2 {
				vec := archsimd.Int64x2{}
				for i := 0; i < 2; i++ {
					vec = vec.SetElem(uint8(i), int64(int64(i)*1000000-1500000))
				}
				return &vec
			},
			expected: 500000,
		},
		{
			name: "all negative values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int64x2 {
				vec := archsimd.Int64x2{}
				for i := 0; i < 2; i++ {
					vec = vec.SetElem(uint8(i), int64(-int64(i)-1))
				}
				return &vec
			},
			expected: -1,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int64x2 {
				return nil
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int64
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMaxInt64x2[int64]()(ro.Empty[*archsimd.Int64x2]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMaxInt64x2[int64]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

func TestClampUint64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		min       uint64
		max       uint64
		expected  []uint64
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []uint64{0, 30000000},
			min:      10000000,
			max:      50000000,
			expected: []uint64{10000000, 30000000},
		},
		{
			name:     "all below min",
			input:    []uint64{0, 5000000},
			min:      10000000,
			max:      50000000,
			expected: []uint64{10000000, 10000000},
		},
		{
			name:     "all above max",
			input:    []uint64{60000000, 70000000},
			min:      10000000,
			max:      50000000,
			expected: []uint64{50000000, 50000000},
		},
		{
			name:      "panic on min > max",
			input:     []uint64{},
			min:       10000000,
			max:       5000000,
			wantPanic: true,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			if tc.wantPanic {
				is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
					_ = ClampUint64x2[uint64](tc.min, tc.max)(ro.Empty[*archsimd.Uint64x2]())
				})
				return
			}

			vec := archsimd.Uint64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint64x2ToScalar[uint64]()(ClampUint64x2[uint64](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		threshold uint64
		expected  []uint64
	}{
		{
			name:      "basic min",
			input:     []uint64{0, 30000000},
			threshold: 40000000,
			expected:  []uint64{40000000, 40000000},
		},
		{
			name:      "all below threshold",
			input:     []uint64{0, 10000000},
			threshold: 40000000,
			expected:  []uint64{40000000, 40000000},
		},
		{
			name:      "all above threshold",
			input:     []uint64{50000000, 60000000},
			threshold: 40000000,
			expected:  []uint64{50000000, 60000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint64x2ToScalar[uint64]()(MinUint64x2[uint64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint64x2(t *testing.T) {
	requireAVX512(t)
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint64
		threshold uint64
		expected  []uint64
	}{
		{
			name:      "basic max",
			input:     []uint64{0, 30000000},
			threshold: 10000000,
			expected:  []uint64{0, 10000000},
		},
		{
			name:      "all below threshold",
			input:     []uint64{0, 5000000},
			threshold: 10000000,
			expected:  []uint64{0, 5000000},
		},
		{
			name:      "all above threshold",
			input:     []uint64{20000000, 30000000},
			threshold: 10000000,
			expected:  []uint64{10000000, 10000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint64x2ToScalar[uint64]()(MaxUint64x2[uint64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint64x2(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Uint64x2
		expected uint64
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Uint64x2 {
				vec := archsimd.Uint64x2{}
				vec = vec.SetElem(0, 60000000)
				vec = vec.SetElem(1, 40000000)
				return &vec
			},
			expected: 40000000,
		},
		{
			name: "all same value",
			setupVec: func() *archsimd.Uint64x2 {
				vec := archsimd.Uint64x2{}
				vec = vec.SetElem(0, 5000000)
				vec = vec.SetElem(1, 5000000)
				return &vec
			},
			expected: 5000000,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Uint64x2 {
				return nil
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []uint64
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMinUint64x2[uint64]()(ro.Empty[*archsimd.Uint64x2]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMinUint64x2[uint64]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint64x2(t *testing.T) {
	requireAVX512(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Uint64x2
		expected uint64
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Uint64x2 {
				vec := archsimd.Uint64x2{}
				vec = vec.SetElem(0, 60000000)
				vec = vec.SetElem(1, 40000000)
				return &vec
			},
			expected: 60000000,
		},
		{
			name: "all same value",
			setupVec: func() *archsimd.Uint64x2 {
				vec := archsimd.Uint64x2{}
				vec = vec.SetElem(0, 5000000)
				vec = vec.SetElem(1, 5000000)
				return &vec
			},
			expected: 5000000,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Uint64x2 {
				return nil
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []uint64
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMaxUint64x2[uint64]()(ro.Empty[*archsimd.Uint64x2]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMaxUint64x2[uint64]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}
