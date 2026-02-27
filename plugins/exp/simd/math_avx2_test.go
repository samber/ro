//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"math"
	"testing"

	"simd/archsimd"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// TestInt8x32 tests AVX2 Int8x32 operators

func TestAddInt8x32(t *testing.T) {
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			addend: 5,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 15
				}
				return v
			}(),
		},
		{
			name: "overflow wraps around",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -106
				}
				return v
			}(),
		},
		{
			name: "underflow wraps around",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			addend: -50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 106
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = int8(i + 1)
					} else {
						v[i] = -int8(i + 1)
					}
				}
				return v
			}(),
			addend: 10,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = int8(i + 1 + 10)
					} else {
						v[i] = -int8(i+1) + 10
					}
				}
				return v
			}(),
		},
		{
			name: "all zeros",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
			addend: 5,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
		},
		{
			name: "boundary values alternating",
			input: func() []int8 {
				v := make([]int8, 32)
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
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = -128 // 127 + 1 = -128 (overflow)
					} else {
						v[i] = -127 // -128 + 1 = -127
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(AddInt8x32[int8](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt8x32(t *testing.T) {
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			subtrahend: 10,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 40
				}
				return v
			}(),
		},
		{
			name: "underflow wraps around",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			subtrahend: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -40
				}
				return v
			}(),
		},
		{
			name: "overflow wraps around (negative - negative)",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			subtrahend: -50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -50
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 100
					} else {
						v[i] = -100
					}
				}
				return v
			}(),
			subtrahend: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 50
					} else {
						v[i] = 106 // -100 - 50 = 106
					}
				}
				return v
			}(),
		},
		{
			name: "boundary values alternating",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			subtrahend: 1,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 126
					} else {
						v[i] = 127 // -128 - 1 = 127
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(SubInt8x32[int8](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		min      int8
		max      int8
		expected []int8
	}{
		{
			name: "basic clamp",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			min: -10,
			max: 10,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			min: -50,
			max: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -50
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			min: -50,
			max: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "all in range",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i)
				}
				return v
			}(),
			min: -10,
			max: 40,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i)
				}
				return v
			}(),
		},
		{
			name: "min equals max",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			min: 0,
			max: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "negative range",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			min: -70,
			max: -30,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
					if val < -70 {
						v[i] = -70
					} else if val > -30 {
						v[i] = -30
					} else {
						v[i] = int8(val)
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(ClampInt8x32[int8](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}

	// Test panic behavior
	t.Run("panic on min > max", func(t *testing.T) {
		is := assert.New(t)
		is.PanicsWithError("simd.Clamp: lower must be less than or equal to upper", func() {
			_ = ClampInt8x32[int8](100, 50)(ro.Empty[*archsimd.Int8x32]())
		})
	})
}

func TestMinInt8x32(t *testing.T) {
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
		},
		{
			name: "negative threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: -50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
					if val < -50 {
						v[i] = -50
					} else {
						v[i] = int8(val)
					}
				}
				return v
			}(),
		},
		{
			name: "positive threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
					if val < 50 {
						v[i] = 50
					} else {
						v[i] = int8(val)
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(MinInt8x32[int8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt8x32(t *testing.T) {
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "negative threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: -50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
					if val > -50 {
						v[i] = -50
					} else {
						v[i] = int8(val)
					}
				}
				return v
			}(),
		},
		{
			name: "positive threshold",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			threshold: 50,
			expected: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					val := i - 16
					if val > 50 {
						v[i] = 50
					} else {
						v[i] = int8(val)
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int8x32ToScalar[int8]()(MaxInt8x32[int8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceSumInt8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    any
		expected int8
		wantErr  string
	}{
		{
			name: "all ones",
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
			name: "all twos",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 2
				}
				return v
			}(),
			expected: 64,
		},
		{
			name: "overflow wraps around",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			expected: 64, // 32 * 10 = 320, 320 - 256 = 64 (wrap around in int8)
		},
		{
			name: "mixed positive and negative",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 10
					} else {
						v[i] = -5
					}
				}
				return v
			}(),
			expected: 80, // 16 * 10 + 16 * (-5) = 160 - 80 = 80
		},
		{
			name: "all negative",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -1
				}
				return v
			}(),
			expected: -32,
		},
		{
			name: "alternating extremes",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 127
					} else {
						v[i] = -128
					}
				}
				return v
			}(),
			expected: -16, // 16 * 127 + 16 * (-128) = 2032 - 2048 = -16
		},
		{
			name: "sequence 0-31",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i)
				}
				return v
			}(),
			expected: -16, // sum of 0-31 = 496, 496 - 256 = 240, which is -16 in int8 (256 - 16 = 240)
		},
		{
			name: "all zeros",
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
			name:     "custom int8 type alias",
			input:    []myInt8{1, 2, 3, 4, 5},
			expected: 15,
		},
		{
			name:     "error propagation",
			input:    ro.Throw[int8](assert.AnError),
			expected: 0,
			wantErr:  assert.AnError.Error(),
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
				vec := ScalarToInt8x32[int8]()(ro.Just(v...))
				result, err = ro.Collect(ReduceSumInt8x32[int8]()(vec))
			case []myInt8:
				vec := ScalarToInt8x32[myInt8]()(ro.Just(v...))
				customResult, customErr := ro.Collect(ReduceSumInt8x32[myInt8]()(vec))
				err = customErr
				result = make([]int8, len(customResult))
				for i, val := range customResult {
					result[i] = int8(val)
				}
			case ro.Observable[int8]:
				vec := ScalarToInt8x32[int8]()(v)
				result, err = ro.Collect(ReduceSumInt8x32[int8]()(vec))
			}

			if tc.wantErr != "" {
				is.EqualError(err, tc.wantErr)
				is.Empty(result)
			} else {
				is.NoError(err)
				is.Equal([]int8{tc.expected}, result)
			}
		})
	}
}

func TestReduceMinInt8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		expected int8
	}{
		{
			name: "basic range",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			expected: -16,
		},
		{
			name: "all positive",
			input: func() []int8 {
				v := make([]int8, 32)
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
				v := make([]int8, 32)
				for i := range v {
					v[i] = -int8(i + 1)
				}
				return v
			}(),
			expected: -32,
		},
		{
			name: "alternating",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 100
					} else {
						v[i] = -100
					}
				}
				return v
			}(),
			expected: -100,
		},
		{
			name: "all same",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			expected: 50,
		},
		{
			name: "with extremes",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i == 0 {
						v[i] = -128
					} else if i == 1 {
						v[i] = 127
					} else {
						v[i] = 0
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt8x32[int8]()(vec))

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int8
		expected int8
	}{
		{
			name: "basic range",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i - 16)
				}
				return v
			}(),
			expected: 15,
		},
		{
			name: "all positive",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = int8(i + 1)
				}
				return v
			}(),
			expected: 32,
		},
		{
			name: "all negative",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -int8(i + 1)
				}
				return v
			}(),
			expected: -1,
		},
		{
			name: "alternating",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i%2 == 0 {
						v[i] = 100
					} else {
						v[i] = -100
					}
				}
				return v
			}(),
			expected: 100,
		},
		{
			name: "all same",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					v[i] = -50
				}
				return v
			}(),
			expected: -50,
		},
		{
			name: "with extremes",
			input: func() []int8 {
				v := make([]int8, 32)
				for i := range v {
					if i == 0 {
						v[i] = -128
					} else if i == 1 {
						v[i] = 127
					} else {
						v[i] = 0
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

			vec := ScalarToInt8x32[int8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt8x32[int8]()(vec))

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

// TestInt16x16 tests AVX2 Int16x16 operators

func TestReduceSumInt16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name:     "sequential 1-16",
			input:    []int16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: 136,
		},
		{
			name:     "all zeros",
			input:    make([]int16, 16),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 16,
		},
		{
			name:     "mixed positive and negative",
			input:    []int16{-100, 200, -300, 400, -500, 600, -700, 800, 100, -200, 300, -400, 500, -600, 700, -800},
			expected: 0,
		},
		{
			name:     "all negative",
			input:    []int16{-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16},
			expected: -136,
		},
		{
			name:     "boundary values",
			input:    []int16{32767, -32768, 32767, -32768, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
			expected: 6,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt16x16[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestAddInt16x16(t *testing.T) {
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i + 1)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i + 1)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps around",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 30000
				}
				return v
			}(),
			addend: 10000,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -25536 // 40000 - 65536
				}
				return v
			}(),
		},
		{
			name: "underflow wraps around",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -30000
				}
				return v
			}(),
			addend: -10000,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 25536
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: []int16{-100, 200, -300, 400, -500, 600, -700, 800, 100, -200, 300, -400, 500, -600, 700, -800},
			addend: 50,
			expected: []int16{-50, 250, -250, 450, -450, 650, -650, 850, 150, -150, 350, -350, 550, -550, 750, -750},
		},
		{
			name: "boundary values",
			input: []int16{32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768, 0, 0, 0, 0, 0, 0, 0, 0},
			addend: 1,
			expected: []int16{-32768, -32767, -32768, -32767, -32768, -32767, -32768, -32767, 1, 1, 1, 1, 1, 1, 1, 1},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(AddInt16x16[int16](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt16x16(t *testing.T) {
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
			subtrahend: 100,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 400
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i * 100)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i * 100)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps around",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			subtrahend: 200,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
		},
		{
			name: "overflow (negative - negative)",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -30000
				}
				return v
			}(),
			subtrahend: -10000,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -20000
				}
				return v
			}(),
		},
		{
			name: "boundary values",
			input: []int16{-32768, 32767, -32768, 32767, -32768, 32767, -32768, 32767, 0, 0, 0, 0, 0, 0, 0, 0},
			subtrahend: 1,
			expected: []int16{32767, 32766, 32767, 32766, 32767, 32766, 32767, 32766, -1, -1, -1, -1, -1, -1, -1, -1},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(SubInt16x16[int16](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt16x16(t *testing.T) {
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i*100 - 700)
				}
				return v
			}(),
			min: -500,
			max: 500,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					val := i*100 - 700
					if val < -500 {
						v[i] = -500
					} else if val > 500 {
						v[i] = 500
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = -1000
				}
				return v
			}(),
			min: -500,
			max: 500,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			min: -500,
			max: 500,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 500
				}
				return v
			}(),
		},
		{
			name: "all in range",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i * 10)
				}
				return v
			}(),
			min: -10,
			max: 200,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i * 10)
				}
				return v
			}(),
		},
		{
			name:     "min equals max",
			input:    []int16{-100, 0, 100, -200, 200, -300, 300, -400, 400, -500, 500, 0, 0, 0, 0, 0},
			min:      50,
			max:      50,
			expected: []int16{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
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
					_ = ClampInt16x16[int16](tc.min, tc.max)(ro.Empty[*archsimd.Int16x16]())
				})
				return
			}

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(ClampInt16x16[int16](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt16x16(t *testing.T) {
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i*100 - 700)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					val := i*100 - 700
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
		},
		{
			name: "negative threshold",
			input: []int16{-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, -100, -200, -300, -400, -500},
			threshold: -250,
			expected: []int16{-250, -250, -250, -200, -100, 0, 100, 200, 300, 400, 500, -100, -200, -250, -250, -250},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(MinInt16x16[int16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt16x16(t *testing.T) {
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i*100 - 700)
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					val := i*100 - 700
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
				v := make([]int16, 16)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "positive threshold",
			input: []int16{-500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, -100, -200, -300, -400, -500},
			threshold: 50,
			expected: []int16{-500, -400, -300, -200, -100, 0, 50, 50, 50, 50, 50, -100, -200, -300, -400, -500},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int16x16ToScalar[int16]()(MaxInt16x16[int16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name: "mixed values",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i*2 - 10)
				}
				return v
			}(),
			expected: -10,
		},
		{
			name: "all positive - detects accumulator init bug",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i + 1)
				}
				return v
			}(),
			expected: 1,
		},
		{
			name: "all negative",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -int16(i + 1)
				}
				return v
			}(),
			expected: -16,
		},
		{
			name: "boundary values",
			input: []int16{32767, -32768, 100, -100, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8},
			expected: -32768,
		},
		{
			name: "single element range",
			input: []int16{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
			expected: 42,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt16x16[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name: "mixed values",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i*2 - 10)
				}
				return v
			}(),
			expected: 20,
		},
		{
			name: "all negative - detects accumulator init bug",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = -int16(i + 1)
				}
				return v
			}(),
			expected: -1,
		},
		{
			name: "all positive",
			input: func() []int16 {
				v := make([]int16, 16)
				for i := range v {
					v[i] = int16(i + 1)
				}
				return v
			}(),
			expected: 16,
		},
		{
			name: "boundary values",
			input: []int16{-32768, 32767, -100, 100, 0, 0, 0, 0, -1, -2, -3, -4, -5, -6, -7, -8},
			expected: 32767,
		},
		{
			name: "single element range",
			input: []int16{-42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42, -42},
			expected: -42,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt16x16[int16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt16x16[int16]()(vec))

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

// TestUint8x32 tests AVX2 Uint8x32 operators

func TestReduceSumUint8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential 1-32",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i + 1)
				}
				return v
			}(),
			expected: 16, // 528 % 256
		},
		{
			name:     "all zeros",
			input:    make([]uint8, 32),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 32,
		},
		{
			name: "overflow wraps",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			expected: 64, // 320 % 256
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint8x32[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

func TestAddUint8x32(t *testing.T) {
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
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			addend: 50,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
			addend: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 44 // 300 % 256
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

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(AddUint8x32[uint8](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint8x32(t *testing.T) {
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
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
			subtrahend: 50,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 150
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 5)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 5)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			subtrahend: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 206 // -50 as uint8
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

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(SubUint8x32[uint8](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint8x32(t *testing.T) {
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
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 10)
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := 0; i < 32; i++ {
					val := uint8(i * 10)
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
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 250
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
		},
		{
			name: "all in range",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(100 + i)
				}
				return v
			}(),
			min: 50,
			max: 200,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					val := uint8(100 + i)
					if val > 200 {
						v[i] = 200
					} else {
						v[i] = val
					}
				}
				return v
			}(),
		},
		{
			name: "min equals max",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i)
				}
				return v
			}(),
			min: 100,
			max: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 100
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
					_ = ClampUint8x32[uint8](tc.min, tc.max)(ro.Empty[*archsimd.Uint8x32]())
				})
				return
			}

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(ClampUint8x32[uint8](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name: "basic min - MinUint8x32 keeps larger of value and threshold",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 10)
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := 0; i < 32; i++ {
					val := uint8(i * 10)
					if val < 100 {
						v[i] = 100
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
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
		},
		{
			name:      "threshold zero",
			input:     []uint8{0, 1, 50, 100, 150, 200, 250, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			threshold: 0,
			expected:  []uint8{0, 1, 50, 100, 150, 200, 250, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(MinUint8x32[uint8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name: "basic max - MaxUint8x32 keeps smaller of value and threshold",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 10)
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := 0; i < 32; i++ {
					val := uint8(i * 10)
					if val > 100 {
						v[i] = 100
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
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 50
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 200
				}
				return v
			}(),
			threshold: 100,
			expected: func() []uint8 {
				v := make([]uint8, 32)
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

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint8x32ToScalar[uint8]()(MaxUint8x32[uint8](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential values",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 10)
				}
				return v
			}(),
			expected: 0,
		},
		{
			name: "all same value",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 42
				}
				return v
			}(),
			expected: 42,
		},
		{
			name: "min at end",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 100
				}
				v[31] = 1
				return v
			}(),
			expected: 1,
		},
		{
			name: "boundary values",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 255
				}
				v[0] = 0
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

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint8x32[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint8x32(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name: "sequential values - max 250 at index 25",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = uint8(i * 10)
				}
				return v
			}(),
			expected: 250,
		},
		{
			name: "all same value",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 42
				}
				return v
			}(),
			expected: 42,
		},
		{
			name: "max at start",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 1
				}
				v[0] = 255
				return v
			}(),
			expected: 255,
		},
		{
			name: "all max",
			input: func() []uint8 {
				v := make([]uint8, 32)
				for i := range v {
					v[i] = 255
				}
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

			vec := ScalarToUint8x32[uint8]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint8x32[uint8]()(vec))

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

// TestUint16x16 tests AVX2 Uint16x16 operators

func TestReduceSumUint16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name:     "sequential 1-16",
			input:    []uint16{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: 136,
		},
		{
			name: "all zeros",
			input: func() []uint16 {
				v := make([]uint16, 16)
				return v
			}(),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 16,
		},
		{
			name: "overflow wraps",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			expected: 13568, // 16*50000 % 65536
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint16x16[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

func TestAddUint16x16(t *testing.T) {
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			addend: 500,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1500
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 100)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 100)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 60000
				}
				return v
			}(),
			addend: 10000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 4464 // 70000 - 65536
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(AddUint16x16[uint16](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint16x16(t *testing.T) {
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			subtrahend: 1000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 4000
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 500)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 500)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			subtrahend: 2000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 64536 // -1000 as uint16
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(SubUint16x16[uint16](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint16x16(t *testing.T) {
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 1000)
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := 0; i < 16; i++ {
					val := uint16(i * 1000)
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
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 20000
				}
				return v
			}(),
			min: 5000,
			max: 10000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 10000
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
					_ = ClampUint16x16[uint16](tc.min, tc.max)(ro.Empty[*archsimd.Uint16x16]())
				})
				return
			}

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(ClampUint16x16[uint16](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint16x16(t *testing.T) {
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 1000)
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := 0; i < 16; i++ {
					val := uint16(i * 1000)
					if val < 5000 {
						v[i] = 5000
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 10000
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 10000
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(MinUint16x16[uint16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint16x16(t *testing.T) {
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 1000)
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := 0; i < 16; i++ {
					val := uint16(i * 1000)
					if val > 5000 {
						v[i] = 5000
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 10000
				}
				return v
			}(),
			threshold: 5000,
			expected: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 5000
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint16x16ToScalar[uint16]()(MaxUint16x16[uint16](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name: "sequential values",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 1000)
				}
				return v
			}(),
			expected: 0,
		},
		{
			name: "all same",
			input: func() []uint16 {
				v := make([]uint16, 16)
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
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 10000
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint16x16[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint16x16(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name: "sequential values",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = uint16(i * 1000)
				}
				return v
			}(),
			expected: 15000,
		},
		{
			name: "all same",
			input: func() []uint16 {
				v := make([]uint16, 16)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			expected: 5000,
		},
		{
			name: "max at start",
			input: func() []uint16 {
				v := make([]uint16, 16)
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

			vec := ScalarToUint16x16[uint16]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint16x16[uint16]()(vec))

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

// TestUint32x8 tests AVX2 Uint32x8 operators

func TestReduceSumUint32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential 1-8",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i+1) * 1000
				}
				return v
			}(),
			expected: 36000,
		},
		{
			name:     "all zeros",
			input:    make([]uint32, 8),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 8,
		},
		{
			name: "overflow wraps",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1000000000
				}
				return v
			}(),
			expected: 3705032704, // 8e9 % 2^32
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint32x8[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

func TestAddUint32x8(t *testing.T) {
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 10000
				}
				return v
			}(),
			addend: 5000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 15000
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
			addend: 0,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 1000)
				}
				return v
			}(),
		},
		{
			name: "overflow wraps",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 4000000000
				}
				return v
			}(),
			addend: 500000000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(AddUint32x8[uint32](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint32x8(t *testing.T) {
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			subtrahend: 10000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 40000
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 5000)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 5000)
				}
				return v
			}(),
		},
		{
			name: "underflow wraps",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			subtrahend: 500,
			expected: func() []uint32 {
				v := make([]uint32, 8)
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(SubUint32x8[uint32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint32x8(t *testing.T) {
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 10000)
				}
				return v
			}(),
			min: 20000,
			max: 50000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := 0; i < 8; i++ {
					val := uint32(i * 10000)
					if val < 20000 {
						v[i] = 20000
					} else if val > 50000 {
						v[i] = 50000
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			min: 20000,
			max: 50000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 20000
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 100000
				}
				return v
			}(),
			min: 20000,
			max: 50000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
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
					_ = ClampUint32x8[uint32](tc.min, tc.max)(ro.Empty[*archsimd.Uint32x8]())
				})
				return
			}

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(ClampUint32x8[uint32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint32x8(t *testing.T) {
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 10000)
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := 0; i < 8; i++ {
					val := uint32(i * 10000)
					if val < 30000 {
						v[i] = 30000
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 30000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(MinUint32x8[uint32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint32x8(t *testing.T) {
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 10000)
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := 0; i < 8; i++ {
					val := uint32(i * 10000)
					if val > 30000 {
						v[i] = 30000
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			threshold: 30000,
			expected: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 30000
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint32x8ToScalar[uint32]()(MaxUint32x8[uint32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential values",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 10000)
				}
				return v
			}(),
			expected: 0,
		},
		{
			name: "all same",
			input: func() []uint32 {
				v := make([]uint32, 8)
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
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 100000
				}
				v[7] = 1
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinUint32x8[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name: "sequential values",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = uint32(i * 10000)
				}
				return v
			}(),
			expected: 70000,
		},
		{
			name: "all same",
			input: func() []uint32 {
				v := make([]uint32, 8)
				for i := range v {
					v[i] = 50000
				}
				return v
			}(),
			expected: 50000,
		},
		{
			name: "max at start",
			input: func() []uint32 {
				v := make([]uint32, 8)
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

			vec := ScalarToUint32x8[uint32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxUint32x8[uint32]()(vec))

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

// TestFloat32x8 tests AVX2 Float32x8 operators

func TestAddFloat32x8(t *testing.T) {
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 1.5
				}
				return v
			}(),
			addend: 2.5,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 4.0
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
			addend: 0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
		},
		{
			name: "negative values",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -10.5
				}
				return v
			}(),
			addend: 5.5,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -5.0
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []float32 {
				v := make([]float32, 8)
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
				v := make([]float32, 8)
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 1000000.0
				}
				return v
			}(),
			addend: 500000.0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 1500000.0
				}
				return v
			}(),
		},
		{
			name: "small values",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 0.0001
				}
				return v
			}(),
			addend: 0.0002,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 0.0003
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

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(AddFloat32x8[float32](tc.addend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat32x8(t *testing.T) {
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5.5
				}
				return v
			}(),
			subtrahend: 2.5,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 3.0
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i)
				}
				return v
			}(),
		},
		{
			name: "negative result",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5.0
				}
				return v
			}(),
			subtrahend: 10.0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -5.0
				}
				return v
			}(),
		},
		{
			name: "negative values",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -10.0
				}
				return v
			}(),
			subtrahend: 5.0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -15.0
				}
				return v
			}(),
		},
		{
			name: "mixed positive and negative",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					if i%2 == 0 {
						v[i] = 10.0
					} else {
						v[i] = -10.0
					}
				}
				return v
			}(),
			subtrahend: 5.0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					if i%2 == 0 {
						v[i] = 5.0
					} else {
						v[i] = -15.0
					}
				}
				return v
			}(),
		},
		{
			name: "large values",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 1000000.0
				}
				return v
			}(),
			subtrahend: 500000.0,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 500000.0
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

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(SubFloat32x8[float32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat32x8(t *testing.T) {
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i * 5)
				}
				return v
			}(),
			min: 10,
			max: 20,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := 0; i < 8; i++ {
					val := float32(i * 5)
					if val < 10 {
						v[i] = 10
					} else if val > 20 {
						v[i] = 20
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			min: 10,
			max: 20,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 10
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			min: 10,
			max: 20,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 20
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
					_ = ClampFloat32x8[float32](tc.min, tc.max)(ro.Empty[*archsimd.Float32x8]())
				})
				return
			}

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(ClampFloat32x8[float32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat32x8(t *testing.T) {
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i * 5)
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := 0; i < 8; i++ {
					val := float32(i * 5)
					if val < 15 {
						v[i] = 15
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 15
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
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

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(MinFloat32x8[float32](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat32x8(t *testing.T) {
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i * 5)
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := 0; i < 8; i++ {
					val := float32(i * 5)
					if val > 15 {
						v[i] = 15
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
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 15,
			expected: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 15
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

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float32x8ToScalar[float32]()(MaxFloat32x8[float32](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceSumFloat32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name: "sequential 1-8 * 1.5",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i+1) * 1.5
				}
				return v
			}(),
			expected: 54,
		},
		{
			name:     "all zeros",
			input:    make([]float32, 8),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 8,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8},
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

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumFloat32x8[float32]()(vec))

			is.NoError(err)
			if math.IsNaN(float64(tc.expected)) {
				is.True(math.IsNaN(float64(result[0])))
			} else if math.IsInf(float64(tc.expected), 0) {
				is.Equal(math.IsInf(float64(tc.expected), 1), math.IsInf(float64(result[0]), 1))
				is.Equal(math.IsInf(float64(tc.expected), -1), math.IsInf(float64(result[0]), -1))
			} else {
				is.InDelta(tc.expected, result[0], 0.0001)
			}
		})
	}
}

func TestReduceMinFloat32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name: "sequential 0-35",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i * 5)
				}
				return v
			}(),
			expected: 0,
		},
		{
			name: "all same",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 5.5
				}
				return v
			}(),
			expected: 5.5,
		},
		{
			name: "min at end",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = 100
				}
				v[7] = -5
				return v
			}(),
			expected: -5,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8},
			expected: float32(math.NaN()),
		},
		{
			name:     "negative Inf is min",
			input:    []float32{1, 2, float32(math.Inf(-1)), 4, 5, 6, 7, 8},
			expected: float32(math.Inf(-1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinFloat32x8[float32]()(vec))

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

func TestReduceMaxFloat32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name: "sequential 0-35",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = float32(i * 5)
				}
				return v
			}(),
			expected: 35,
		},
		{
			name: "all same",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -5.5
				}
				return v
			}(),
			expected: -5.5,
		},
		{
			name: "max at start",
			input: func() []float32 {
				v := make([]float32, 8)
				for i := range v {
					v[i] = -100
				}
				v[0] = 1000
				return v
			}(),
			expected: 1000,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4, 5, 6, 7, 8},
			expected: float32(math.NaN()),
		},
		{
			name:     "positive Inf is max",
			input:    []float32{1, 2, float32(math.Inf(1)), 4, 5, 6, 7, 8},
			expected: float32(math.Inf(1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat32x8[float32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxFloat32x8[float32]()(vec))

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

// TestFloat64x4 tests AVX2 Float64x4 operators

func TestReduceSumFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "sequential values",
			input:    []float64{1.5, 2.5, 3.5, 4.5},
			expected: 12,
		},
		{
			name:     "all zeros",
			input:    []float64{0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "negative values",
			input:    []float64{-1.0, -2.0, -3.0, -4.0},
			expected: -10,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, 2, math.NaN(), 4},
			expected: math.NaN(),
		},
		{
			name:     "positive Inf",
			input:    []float64{1, 2, math.Inf(1), 4},
			expected: math.Inf(1),
		},
		{
			name:     "negative Inf",
			input:    []float64{math.Inf(-1), 2, 3, 4},
			expected: math.Inf(-1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumFloat64x4[float64]()(vec))

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

func TestAddFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		addend   float64
		expected []float64
	}{
		{
			name:     "basic addition",
			input:    []float64{1.5, 1.5, 1.5, 1.5},
			addend:   2.5,
			expected: []float64{4.0, 4.0, 4.0, 4.0},
		},
		{
			name:     "add zero",
			input:    []float64{1.0, 2.0, 3.0, 4.0},
			addend:   0,
			expected: []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:     "negative values",
			input:    []float64{-10.0, -10.0, -10.0, -10.0},
			addend:   5.0,
			expected: []float64{-5.0, -5.0, -5.0, -5.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(AddFloat64x4[float64](tc.addend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		input      []float64
		subtrahend float64
		expected   []float64
	}{
		{
			name:       "basic subtraction",
			input:      []float64{10.5, 10.5, 10.5, 10.5},
			subtrahend: 3.5,
			expected:   []float64{7.0, 7.0, 7.0, 7.0},
		},
		{
			name:       "subtract zero",
			input:      []float64{1.0, 2.0, 3.0, 4.0},
			subtrahend: 0,
			expected:   []float64{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:       "negative result",
			input:      []float64{5.0, 5.0, 5.0, 5.0},
			subtrahend: 10.0,
			expected:   []float64{-5.0, -5.0, -5.0, -5.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(SubFloat64x4[float64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat64x4(t *testing.T) {
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
			input:    []float64{-5.5, 0.0, 5.5, 10.5},
			min:      -2.5,
			max:      7.5,
			expected: []float64{-2.5, 0.0, 5.5, 7.5},
		},
		{
			name:     "all below min",
			input:    []float64{-10.0, -10.0, -10.0, -10.0},
			min:      -2.5,
			max:      7.5,
			expected: []float64{-2.5, -2.5, -2.5, -2.5},
		},
		{
			name:     "all above max",
			input:    []float64{100.0, 100.0, 100.0, 100.0},
			min:      -2.5,
			max:      7.5,
			expected: []float64{7.5, 7.5, 7.5, 7.5},
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
					_ = ClampFloat64x4[float64](tc.min, tc.max)(ro.Empty[*archsimd.Float64x4]())
				})
				return
			}

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(ClampFloat64x4[float64](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic min",
			input:     []float64{-5.5, 0.0, 5.5, 10.5},
			threshold: 2.5,
			expected:  []float64{2.5, 2.5, 5.5, 10.5},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10.0, -5.0, 0.0, 1.0},
			threshold: 2.5,
			expected:  []float64{2.5, 2.5, 2.5, 2.5},
		},
		{
			name:      "all above threshold",
			input:     []float64{10.0, 20.0, 30.0, 40.0},
			threshold: 2.5,
			expected:  []float64{10.0, 20.0, 30.0, 40.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(MinFloat64x4[float64](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic max",
			input:     []float64{-5.5, 0.0, 5.5, 10.5},
			threshold: 5.5,
			expected:  []float64{-5.5, 0.0, 5.5, 5.5},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10.0, -5.0, 0.0, 1.0},
			threshold: 5.5,
			expected:  []float64{-10.0, -5.0, 0.0, 1.0},
		},
		{
			name:      "all above threshold",
			input:     []float64{10.0, 20.0, 30.0, 40.0},
			threshold: 5.5,
			expected:  []float64{5.5, 5.5, 5.5, 5.5},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Float64x4ToScalar[float64]()(MaxFloat64x4[float64](tc.threshold)(vec)),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceMinFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "mixed values",
			input:    []float64{3.5, 1.5, 4.5, 2.5},
			expected: 1.5,
		},
		{
			name:     "all same",
			input:    []float64{5.0, 5.0, 5.0, 5.0},
			expected: 5.0,
		},
		{
			name:     "min at end",
			input:    []float64{10.0, 20.0, 30.0, -5.0},
			expected: -5.0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinFloat64x4[float64]()(vec))

			is.NoError(err)
			is.InDelta(tc.expected, result[0], 0.0001)
		})
	}
}

func TestReduceMaxFloat64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "mixed values",
			input:    []float64{3.5, 1.5, 4.5, 2.5},
			expected: 4.5,
		},
		{
			name:     "all same",
			input:    []float64{5.0, 5.0, 5.0, 5.0},
			expected: 5.0,
		},
		{
			name:     "max at start",
			input:    []float64{100.0, 1.0, 2.0, 3.0},
			expected: 100.0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToFloat64x4[float64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxFloat64x4[float64]()(vec))

			is.NoError(err)
			is.InDelta(tc.expected, result[0], 0.0001)
		})
	}
}

// TestInt64x4 tests AVX2 Int64x4 operators (Add/Sub only - Min/Max/Clamp require AVX-512)

func TestReduceSumInt64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "sequential values",
			input:    []int64{1000000, 2000000, 3000000, 4000000},
			expected: 10000000,
		},
		{
			name:     "all zeros",
			input:    []int64{0, 0, 0, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt64x4[int64]()(vec))

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

func TestAddInt64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int64
		addend   int64
		expected []int64
	}{
		{
			name:     "basic addition",
			input:    []int64{1000000, 1000000, 1000000, 1000000},
			addend:   500000,
			expected: []int64{1500000, 1500000, 1500000, 1500000},
		},
		{
			name:     "add zero",
			input:    []int64{1000000, 2000000, 3000000, 4000000},
			addend:   0,
			expected: []int64{1000000, 2000000, 3000000, 4000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(AddInt64x4[int64](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		input      []int64
		subtrahend int64
		expected   []int64
	}{
		{
			name:       "basic subtraction",
			input:      []int64{5000000, 5000000, 5000000, 5000000},
			subtrahend: 2000000,
			expected:   []int64{3000000, 3000000, 3000000, 3000000},
		},
		{
			name:       "subtract zero",
			input:      []int64{1000000, 2000000, 3000000, 4000000},
			subtrahend: 0,
			expected:   []int64{1000000, 2000000, 3000000, 4000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt64x4[int64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int64x4ToScalar[int64]()(SubInt64x4[int64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// TestUint64x4 tests AVX2 Uint64x4 operators (Add/Sub only - Min/Max/Clamp require AVX-512)

func TestReduceSumUint64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "sequential values",
			input:    []uint64{1000000, 2000000, 3000000, 4000000},
			expected: 10000000,
		},
		{
			name:     "all zeros",
			input:    []uint64{0, 0, 0, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumUint64x4[uint64]()(vec))

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

func TestAddUint64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []uint64
		addend   uint64
		expected []uint64
	}{
		{
			name: "basic addition",
			input: func() []uint64 {
				v := make([]uint64, 4)
				for i := range v {
					v[i] = 5000000
				}
				return v
			}(),
			addend: 2000000,
			expected: func() []uint64 {
				v := make([]uint64, 4)
				for i := range v {
					v[i] = 7000000
				}
				return v
			}(),
		},
		{
			name:     "add zero",
			input:    []uint64{1000000, 2000000, 3000000, 4000000},
			addend:   0,
			expected: []uint64{1000000, 2000000, 3000000, 4000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(AddUint64x4[uint64](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint64x4(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name       string
		input      []uint64
		subtrahend uint64
		expected   []uint64
	}{
		{
			name: "basic subtraction",
			input: func() []uint64 {
				v := make([]uint64, 4)
				for i := range v {
					v[i] = 10000000
				}
				return v
			}(),
			subtrahend: 5000000,
			expected: func() []uint64 {
				v := make([]uint64, 4)
				for i := range v {
					v[i] = 5000000
				}
				return v
			}(),
		},
		{
			name:       "subtract zero",
			input:      []uint64{1000000, 2000000, 3000000, 4000000},
			subtrahend: 0,
			expected:   []uint64{1000000, 2000000, 3000000, 4000000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToUint64x4[uint64]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Uint64x4ToScalar[uint64]()(SubUint64x4[uint64](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// TestInt32x8 tests AVX2 Int32x8 operators

func TestReduceSumInt32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential 1-8 * 1000",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i+1) * 1000
				}
				return v
			}(),
			expected: 36000,
		},
		{
			name:     "all zeros",
			input:    make([]int32, 8),
			expected: 0,
		},
		{
			name: "all ones",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: 8,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceSumInt32x8[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

func TestAddInt32x8(t *testing.T) {
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
				v := make([]int32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			addend: 500,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 1500
				}
				return v
			}(),
		},
		{
			name: "add zero",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			addend: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
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

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(AddInt32x8[int32](tc.addend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt32x8(t *testing.T) {
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
				v := make([]int32, 8)
				for i := range v {
					v[i] = 1000
				}
				return v
			}(),
			subtrahend: 300,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 700
				}
				return v
			}(),
		},
		{
			name: "subtract zero",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			subtrahend: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
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

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(SubInt32x8[int32](tc.subtrahend)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt32x8(t *testing.T) {
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
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			min:      -2,
			max:      2,
			expected: []int32{-2, -2, -2, -1, 0, 1, 2, 2},
		},
		{
			name: "all below min",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -100
				}
				return v
			}(),
			min: -2,
			max: 2,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -2
				}
				return v
			}(),
		},
		{
			name: "all above max",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			min: -2,
			max: 2,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 2
				}
				return v
			}(),
		},
		{
			name:      "panic on min > max",
			input:     []int32{},
			min:       1000,
			max:       500,
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
					_ = ClampInt32x8[int32](tc.min, tc.max)(ro.Empty[*archsimd.Int32x8]())
				})
				return
			}

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(ClampInt32x8[int32](tc.min, tc.max)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt32x8(t *testing.T) {
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
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			threshold: 0,
			expected:  []int32{0, 0, 0, 0, 0, 1, 2, 3},
		},
		{
			name: "all below threshold",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -10
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 0
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
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

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(MinInt32x8[int32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt32x8(t *testing.T) {
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
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			threshold: 0,
			expected:  []int32{-4, -3, -2, -1, 0, 0, 0, 0},
		},
		{
			name: "all below threshold",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -10
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -10
				}
				return v
			}(),
		},
		{
			name: "all above threshold",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 100
				}
				return v
			}(),
			threshold: 0,
			expected: func() []int32 {
				v := make([]int32, 8)
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

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(
				Int32x8ToScalar[int32]()(MaxInt32x8[int32](tc.threshold)(vec)),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential -4 to 3",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			expected: -4,
		},
		{
			name: "all same",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 5000
				}
				return v
			}(),
			expected: 5000,
		},
		{
			name: "min at end",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = 100
				}
				v[7] = -1000
				return v
			}(),
			expected: -1000,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMinInt32x8[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt32x8(t *testing.T) {
	t.Parallel()

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name: "sequential -4 to 3",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = int32(i - 4)
				}
				return v
			}(),
			expected: 3,
		},
		{
			name: "all same",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -500
				}
				return v
			}(),
			expected: -500,
		},
		{
			name: "max at start",
			input: func() []int32 {
				v := make([]int32, 8)
				for i := range v {
					v[i] = -100
				}
				v[0] = 10000
				return v
			}(),
			expected: 10000,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := ScalarToInt32x8[int32]()(ro.Just(tc.input...))
			result, err := ro.Collect(ReduceMaxInt32x8[int32]()(vec))

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}
