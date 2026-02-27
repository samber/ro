//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"math"
	"testing"

	"simd/archsimd"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// ReduceInt8x16 tests

func TestReduceSumInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    any
		expected any
		wantErr  string
	}{
		{
			name:     "empty input",
			input:    []int8{},
			expected: []int8{0},
		},
		{
			name:     "single value",
			input:    []int8{5},
			expected: []int8{5},
		},
		{
			name:     "full buffer (16 values)",
			input:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: []int8{-120}, // sum of 1-16 wrapped: 136 - 256 = -120
		},
		{
			name:     "overflow buffer (18 values)",
			input:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18},
			expected: []int8{-85}, // 171 mod 256 = -85 in int8 two's complement
		},
		{
			name:     "negative values",
			input:    []int8{-1, -2, -3, -4, -5},
			expected: []int8{-15},
		},
		{
			name:     "mixed positive and negative",
			input:    []int8{10, -5, 3, -2},
			expected: []int8{6},
		},
		{
			name:     "partial buffer (15 values)",
			input:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expected: []int8{120}, // sum of 1-15
		},
		{
			name:     "small partial buffer (5 values)",
			input:    []int8{10, 20, 30, 40, 50},
			expected: []int8{-106}, // sum 150 wrapped: 150 - 256 = -106
		},
		{
			name:     "int8 overflow handling",
			input:    []int8{100, 100, 100},
			expected: []int8{44}, // 300 mod 256
		},
		{
			name: "large input (50 values)",
			input: func() []int8 {
				v := make([]int8, 50)
				for i := range v {
					v[i] = 1
				}
				return v
			}(),
			expected: []int8{50},
		},
		{
			name:     "custom int8 type",
			input:    []myInt8{1, 2, 3, 4, 5},
			expected: []int8{15},
		},
		{
			name:     "error propagation",
			input:    ro.Throw[int8](assert.AnError),
			expected: []int8{},
			wantErr:  assert.AnError.Error(),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var values []int8
			var err error

			switch v := tc.input.(type) {
			case []int8:
				values, err = ro.Collect(
					ReduceSumInt8x16[int8]()(ScalarToInt8x16[int8]()(ro.Just(v...))),
				)
			case []myInt8:
				customValues, customErr := ro.Collect(
					ReduceSumInt8x16[myInt8]()(ScalarToInt8x16[myInt8]()(ro.Just(v...))),
				)
				// Convert myInt8 to int8 for comparison
				values = make([]int8, len(customValues))
				for i, val := range customValues {
					values[i] = int8(val)
				}
				err = customErr
			case ro.Observable[int8]:
				values, err = ro.Collect(
					ReduceSumInt8x16[int8]()(ScalarToInt8x16[int8]()(v)),
				)
			}

			if tc.wantErr != "" {
				is.EqualError(err, tc.wantErr)
				is.Empty(values)
			} else {
				is.NoError(err)
				is.Equal(tc.expected, values)
			}
		})
	}
}

func TestReduceMinInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int8x16
		expected int8
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(i*2-10))
				}
				return &vec
			},
			expected: -10, // min is -10 at index 0
		},
		{
			name: "all positive values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(i+1)) // values 1-16
				}
				return &vec
			},
			expected: 1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "all negative values",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(-int8(i)-1)) // values -1 to -16
				}
				return &vec
			},
			expected: -16,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int8x16 {
				return nil // will use Empty() observable
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int8
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMinInt8x16[int8]()(ro.Empty[*archsimd.Int8x16]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMinInt8x16[int8]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int8x16
		expected int8
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(i*2-10))
				}
				return &vec
			},
			expected: 20, // max is 20 at index 15
		},
		{
			name: "all negative values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(-int8(i)-1)) // values -1 to -16
				}
				return &vec
			},
			expected: -1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "all positive values",
			setupVec: func() *archsimd.Int8x16 {
				vec := archsimd.Int8x16{}
				for i := 0; i < 16; i++ {
					vec = vec.SetElem(uint8(i), int8(i+1)) // values 1-16
				}
				return &vec
			},
			expected: 16,
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int8x16 {
				return nil // will use Empty() observable
			},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var result []int8
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMaxInt8x16[int8]()(ro.Empty[*archsimd.Int8x16]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMaxInt8x16[int8]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int8{tc.expected}, result)
		})
	}
}

// Add/Sub/Clamp/Min/Max tests

func TestAddInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int8
		addend   int8
		expected []int8
	}{
		{
			name:     "basic addition",
			input:    []int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			addend:   2,
			expected: []int8{3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3},
		},
		{
			name:     "overflow wraps around",
			input:    []int8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
			addend:   50,
			expected: []int8{-106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106, -106},
		},
		{
			name:     "underflow wraps around",
			input:    []int8{-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100},
			addend:   -50,
			expected: []int8{106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106},
		},
		{
			name:     "add zero",
			input:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			addend:   0,
			expected: []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			name:     "add to negative values",
			input:    []int8{-10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -125, -126, -127, -128},
			addend:   5,
			expected: []int8{-5, -15, -25, -35, -45, -55, -65, -75, -85, -95, -105, -115, -120, -121, -122, -123}, // -128 + 5 = -123
		},
		{
			name:     "mixed positive and negative",
			input:    []int8{-50, -25, 0, 25, 50, -50, -25, 0, 25, 50, -50, -25, 0, 25, 50, 0},
			addend:   10,
			expected: []int8{-40, -15, 10, 35, 60, -40, -15, 10, 35, 60, -40, -15, 10, 35, 60, 10},
		},
		{
			name:     "all zeros",
			input:    []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			addend:   5,
			expected: []int8{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
		},
		{
			name:     "boundary values",
			input:    []int8{127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128},
			addend:   1,
			expected: []int8{-128, -127, -128, -127, -128, -127, -128, -127, -128, -127, -128, -127, -128, -127, -128, -127},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(AddInt8x16[int8](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []int8
		subtrahend int8
		expected   []int8
	}{
		{
			name:       "basic subtraction",
			input:      []int8{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
			subtrahend: 3,
			expected:   []int8{7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7},
		},
		{
			name:       "underflow wraps around",
			input:      []int8{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
			subtrahend: 50,
			expected:   []int8{-40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40, -40},
		},
		{
			name:       "overflow wraps around (negative - negative)",
			input:      []int8{-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100},
			subtrahend: -50,
			expected:   []int8{-50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50, -50},
		},
		{
			name:       "subtract zero",
			input:      []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			subtrahend: 0,
			expected:   []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		},
		{
			name:       "subtract from negative values",
			input:      []int8{-10, -20, -30, -40, -50, -60, -70, -80, -90, -100, -110, -120, -125, -126, -127, -128},
			subtrahend: -5,
			expected:   []int8{-5, -15, -25, -35, -45, -55, -65, -75, -85, -95, -105, -115, -120, -121, -122, -123},
		},
		{
			name:       "boundary values",
			input:      []int8{-128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127},
			subtrahend: 1,
			expected:   []int8{127, 126, 127, 126, 127, 126, 127, 126, 127, 126, 127, 126, 127, 126, 127, 126},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(SubInt8x16[int8](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int8
		min       int8
		max       int8
		expected  []int8
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
			min:      -5,
			max:      5,
			expected: []int8{-5, -5, -5, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 5, 5},
		},
		{
			name:     "all below min",
			input:    []int8{-128, -100, -50, -30, -20, -15, -10, -10, -10, -10, -10, -10, -10, -10, -10, -10},
			min:      -5,
			max:      5,
			expected: []int8{-5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5, -5},
		},
		{
			name:     "all above max",
			input:    []int8{10, 20, 30, 50, 100, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
			min:      -5,
			max:      5,
			expected: []int8{5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5},
		},
		{
			name:     "all in range",
			input:    []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			min:      -10,
			max:      20,
			expected: []int8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
		{
			name:     "boundary clamp values",
			input:    []int8{127, -128, 100, -100, 50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			min:      -50,
			max:      50,
			expected: []int8{50, -50, 50, -50, 50, -50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:     "negative range",
			input:    []int8{-100, -80, -60, -40, -20, 0, 20, 40, 60, 80, 100, -100, -80, -60, -40, -20},
			min:      -70,
			max:      -30,
			expected: []int8{-70, -70, -60, -40, -30, -30, -30, -30, -30, -30, -30, -70, -70, -60, -40, -30},
		},
		{
			name:     "min equals max",
			input:    []int8{-100, -50, 0, 50, 100, -100, -50, 0, 50, 100, -100, -50, 0, 50, 100, 0},
			min:      0,
			max:      0,
			expected: []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "panic on min > max",
			input:     []int8{},
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
					_ = ClampInt8x16[int8](tc.min, tc.max)(ro.Empty[*archsimd.Int8x16]())
				})
				return
			}

			vec := archsimd.Int8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(ClampInt8x16[int8](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int8
		threshold int8
		expected  []int8
	}{
		{
			name:      "basic min",
			input:     []int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
			threshold: 0,
			expected:  []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7},
		},
		{
			name:      "all below threshold",
			input:     []int8{-100, -50, -30, -20, -10, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
			threshold: 0,
			expected:  []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "all above threshold",
			input:     []int8{10, 20, 30, 50, 100, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
			threshold: 0,
			expected:  []int8{10, 20, 30, 50, 100, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
		},
		{
			name:      "negative threshold",
			input:     []int8{-100, -80, -60, -40, -20, 0, 20, 40, -100, -80, -60, -40, -20, 0, 20, 40},
			threshold: -50,
			expected:  []int8{-50, -50, -50, -40, -20, 0, 20, 40, -50, -50, -50, -40, -20, 0, 20, 40},
		},
		{
			name:      "positive threshold",
			input:     []int8{-100, -50, 0, 50, 100, -100, -50, 0, 50, 100, -100, -50, 0, 50, 100, 0},
			threshold: 50,
			expected:  []int8{50, 50, 50, 50, 100, 50, 50, 50, 50, 100, 50, 50, 50, 50, 100, 50},
		},
		{
			name:      "boundary values",
			input:     []int8{127, -128, 100, -100, 50, -50, 0, 0, 127, -128, 100, -100, 50, -50, 0, 0},
			threshold: 0,
			expected:  []int8{127, 0, 100, 0, 50, 0, 0, 0, 127, 0, 100, 0, 50, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(MinInt8x16[int8](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int8
		threshold int8
		expected  []int8
	}{
		{
			name:      "basic max",
			input:     []int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7},
			threshold: 0,
			expected:  []int8{-8, -7, -6, -5, -4, -3, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "all below threshold",
			input:     []int8{-100, -50, -30, -20, -10, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
			threshold: 0,
			expected:  []int8{-100, -50, -30, -20, -10, -5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		},
		{
			name:      "all above threshold",
			input:     []int8{10, 20, 30, 50, 100, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
			threshold: 0,
			expected:  []int8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "negative threshold",
			input:     []int8{-100, -80, -60, -40, -20, 0, 20, 40, -100, -80, -60, -40, -20, 0, 20, 40},
			threshold: -50,
			expected:  []int8{-100, -80, -60, -50, -50, -50, -50, -50, -100, -80, -60, -50, -50, -50, -50, -50},
		},
		{
			name:      "positive threshold",
			input:     []int8{-100, -50, 0, 50, 100, -100, -50, 0, 50, 100, -100, -50, 0, 50, 100, 0},
			threshold: 50,
			expected:  []int8{-100, -50, 0, 50, 50, -100, -50, 0, 50, 50, -100, -50, 0, 50, 50, 0},
		},
		{
			name:      "boundary values",
			input:     []int8{127, -128, 100, -100, 50, -50, 0, 0, 127, -128, 100, -100, 50, -50, 0, 0},
			threshold: 0,
			expected:  []int8{0, -128, 0, -100, 0, -50, 0, 0, 0, -128, 0, -100, 0, -50, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int8x16ToScalar[int8]()(MaxInt8x16[int8](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// Int16x8 tests

func TestReduceSumInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int16
		expected int16
	}{
		{
			name:     "sequential 1-8",
			input:    []int16{1, 2, 3, 4, 5, 6, 7, 8},
			expected: 36,
		},
		{
			name:     "all zeros",
			input:    []int16{0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "all ones",
			input:    []int16{1, 1, 1, 1, 1, 1, 1, 1},
			expected: 8,
		},
		{
			name:     "mixed positive and negative",
			input:    []int16{-100, 200, -300, 400, -500, 600, -700, 800},
			expected: 400,
		},
		{
			name:     "all negative",
			input:    []int16{-1, -2, -3, -4, -5, -6, -7, -8},
			expected: -36,
		},
		{
			name:     "boundary values",
			input:    []int16{32767, -32768, 32767, -32768, 0, 0, 1, 1},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumInt16x8[int16]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestSubInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []int16
		subtrahend int16
		expected   []int16
	}{
		{
			name:       "basic subtraction",
			input:      []int16{100, 100, 100, 100, 100, 100, 100, 100},
			subtrahend: 30,
			expected:   []int16{70, 70, 70, 70, 70, 70, 70, 70},
		},
		{
			name:       "underflow wraps around",
			input:      []int16{100, 100, 100, 100, 100, 100, 100, 100},
			subtrahend: 200,
			expected:   []int16{-100, -100, -100, -100, -100, -100, -100, -100},
		},
		{
			name:       "overflow wraps around (negative - negative)",
			input:      []int16{-30000, -30000, -30000, -30000, -30000, -30000, -30000, -30000},
			subtrahend: -10000,
			expected:   []int16{-20000, -20000, -20000, -20000, -20000, -20000, -20000, -20000},
		},
		{
			name:       "subtract zero",
			input:      []int16{100, 200, 300, 400, 500, 600, 700, 800},
			subtrahend: 0,
			expected:   []int16{100, 200, 300, 400, 500, 600, 700, 800},
		},
		{
			name:       "mixed positive and negative",
			input:      []int16{-100, 200, -300, 400, -500, 600, -700, 800},
			subtrahend: 50,
			expected:   []int16{-150, 150, -350, 350, -550, 550, -750, 750},
		},
		{
			name:       "boundary values",
			input:      []int16{-32768, 32767, -32768, 32767, -32768, 32767, -32768, 32767},
			subtrahend: 1,
			expected:   []int16{32767, 32766, 32767, 32766, 32767, 32766, 32767, 32766},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(SubInt16x8[int16](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int16
		min       int16
		max       int16
		expected  []int16
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []int16{-35, -25, -15, -5, 5, 15, 25, 35},
			min:      -10,
			max:      20,
			expected: []int16{-10, -10, -10, -5, 5, 15, 20, 20},
		},
		{
			name:     "all below min",
			input:    []int16{-100, -200, -300, -400, -500, -600, -700, -800},
			min:      -50,
			max:      50,
			expected: []int16{-50, -50, -50, -50, -50, -50, -50, -50},
		},
		{
			name:     "all above max",
			input:    []int16{100, 200, 300, 400, 500, 600, 700, 800},
			min:      -50,
			max:      50,
			expected: []int16{50, 50, 50, 50, 50, 50, 50, 50},
		},
		{
			name:     "all in range",
			input:    []int16{10, 20, 30, 40, 50, 60, 70, 80},
			min:      0,
			max:      100,
			expected: []int16{10, 20, 30, 40, 50, 60, 70, 80},
		},
		{
			name:     "boundary clamp values",
			input:    []int16{32767, -32768, 10000, -10000, 5000, -5000, 0, 0},
			min:      -5000,
			max:      5000,
			expected: []int16{5000, -5000, 5000, -5000, 5000, -5000, 0, 0},
		},
		{
			name:     "negative range",
			input:    []int16{-1000, -800, -600, -400, -200, 0, 200, 400},
			min:      -700,
			max:      -300,
			expected: []int16{-700, -700, -600, -400, -300, -300, -300, -300},
		},
		{
			name:     "min equals max",
			input:    []int16{-1000, -500, 0, 500, 1000, -1000, -500, 0},
			min:      0,
			max:      0,
			expected: []int16{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "panic on min > max",
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
					_ = ClampInt16x8[int16](tc.min, tc.max)(ro.Empty[*archsimd.Int16x8]())
				})
				return
			}

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(ClampInt16x8[int16](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int16
		threshold int16
		expected  []int16
	}{
		{
			name:      "basic min",
			input:     []int16{-35, -25, -15, -5, 5, 15, 25, 35},
			threshold: 0,
			expected:  []int16{0, 0, 0, 0, 5, 15, 25, 35},
		},
		{
			name:      "all below threshold",
			input:     []int16{-100, -200, -300, -400, -500, -600, -700, -800},
			threshold: 0,
			expected:  []int16{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "all above threshold",
			input:     []int16{100, 200, 300, 400, 500, 600, 700, 800},
			threshold: 0,
			expected:  []int16{100, 200, 300, 400, 500, 600, 700, 800},
		},
		{
			name:      "negative threshold",
			input:     []int16{-500, -400, -300, -200, -100, 0, 100, 200},
			threshold: -250,
			expected:  []int16{-250, -250, -250, -200, -100, 0, 100, 200},
		},
		{
			name:      "positive threshold",
			input:     []int16{-500, -400, -300, -200, -100, 0, 100, 200},
			threshold: 50,
			expected:  []int16{50, 50, 50, 50, 50, 50, 100, 200},
		},
		{
			name:      "boundary values",
			input:     []int16{32767, -32768, 100, -100, 0, 0, 0, 0},
			threshold: 0,
			expected:  []int16{32767, 0, 100, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(MinInt16x8[int16](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int16
		threshold int16
		expected  []int16
	}{
		{
			name:      "basic max",
			input:     []int16{-35, -25, -15, -5, 5, 15, 25, 35},
			threshold: 0,
			expected:  []int16{-35, -25, -15, -5, 0, 0, 0, 0},
		},
		{
			name:      "all below threshold",
			input:     []int16{-100, -200, -300, -400, -500, -600, -700, -800},
			threshold: 0,
			expected:  []int16{-100, -200, -300, -400, -500, -600, -700, -800},
		},
		{
			name:      "all above threshold",
			input:     []int16{100, 200, 300, 400, 500, 600, 700, 800},
			threshold: 0,
			expected:  []int16{0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:      "negative threshold",
			input:     []int16{-500, -400, -300, -200, -100, 0, 100, 200},
			threshold: -250,
			expected:  []int16{-500, -400, -300, -250, -250, -250, -250, -250},
		},
		{
			name:      "boundary values",
			input:     []int16{-32768, 32767, -100, 100, 0, 0, 0, 0},
			threshold: 0,
			expected:  []int16{-32768, 0, -100, 0, 0, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(MaxInt16x8[int16](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int16x8
		expected int16
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int16x8 {
				vec := archsimd.Int16x8{}
				for i := 0; i < 8; i++ {
					vec = vec.SetElem(uint8(i), int16(i*10-35))
				}
				return &vec
			},
			expected: -35, // min is -35 at index 0
		},
		{
			name: "all positive values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int16x8 {
				vec := archsimd.Int16x8{}
				for i := 0; i < 8; i++ {
					vec = vec.SetElem(uint8(i), int16(i+1)) // values 1-8
				}
				return &vec
			},
			expected: 1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int16x8 {
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

			var result []int16
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMinInt16x8[int16]()(ro.Empty[*archsimd.Int16x8]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMinInt16x8[int16]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int16x8
		expected int16
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int16x8 {
				vec := archsimd.Int16x8{}
				for i := 0; i < 8; i++ {
					vec = vec.SetElem(uint8(i), int16(i*10-35))
				}
				return &vec
			},
			expected: 35, // max is 35 at index 7
		},
		{
			name: "all negative values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int16x8 {
				vec := archsimd.Int16x8{}
				for i := 0; i < 8; i++ {
					vec = vec.SetElem(uint8(i), int16(-int16(i)-1)) // values -1 to -8
				}
				return &vec
			},
			expected: -1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int16x8 {
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

			var result []int16
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMaxInt16x8[int16]()(ro.Empty[*archsimd.Int16x8]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMaxInt16x8[int16]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int16{tc.expected}, result)
		})
	}
}

func TestAddInt16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int16
		addend   int16
		expected []int16
	}{
		{
			name:     "basic addition",
			input:    []int16{100, 100, 100, 100, 100, 100, 100, 100},
			addend:   50,
			expected: []int16{150, 150, 150, 150, 150, 150, 150, 150},
		},
		{
			name:     "overflow wraps around",
			input:    []int16{30000, 30000, 30000, 30000, 30000, 30000, 30000, 30000},
			addend:   10000,
			expected: []int16{-25536, -25536, -25536, -25536, -25536, -25536, -25536, -25536},
		},
		{
			name:     "underflow wraps around",
			input:    []int16{-30000, -30000, -30000, -30000, -30000, -30000, -30000, -30000},
			addend:   -10000,
			expected: []int16{25536, 25536, 25536, 25536, 25536, 25536, 25536, 25536},
		},
		{
			name:     "add zero",
			input:    []int16{100, 200, 300, 400, 500, 600, 700, 800},
			addend:   0,
			expected: []int16{100, 200, 300, 400, 500, 600, 700, 800},
		},
		{
			name:     "mixed positive and negative",
			input:    []int16{-100, 200, -300, 400, -500, 600, -700, 800},
			addend:   50,
			expected: []int16{-50, 250, -250, 450, -450, 650, -650, 850},
		},
		{
			name:     "all zeros",
			input:    []int16{0, 0, 0, 0, 0, 0, 0, 0},
			addend:   100,
			expected: []int16{100, 100, 100, 100, 100, 100, 100, 100},
		},
		{
			name:     "boundary values",
			input:    []int16{32767, -32768, 32767, -32768, 32767, -32768, 32767, -32768},
			addend:   1,
			expected: []int16{-32768, -32767, -32768, -32767, -32768, -32767, -32768, -32767},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int16x8ToScalar[int16]()(AddInt16x8[int16](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// Int32x4 tests

func TestReduceSumInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int32
		expected int32
	}{
		{
			name:     "basic sum",
			input:    []int32{1000, 2000, 3000, 4000},
			expected: 10000,
		},
		{
			name:     "all zeros",
			input:    []int32{0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "mixed positive and negative",
			input:    []int32{-1000, 2000, -3000, 4000},
			expected: 2000,
		},
		{
			name:     "all negative",
			input:    []int32{-1000, -2000, -3000, -4000},
			expected: -10000,
		},
		{
			name:     "boundary values",
			input:    []int32{2147483647, -2147483648, 0, 1},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumInt32x4[int32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

func TestAddInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int32
		addend   int32
		expected []int32
	}{
		{
			name:     "basic addition",
			input:    []int32{1000, 1000, 1000, 1000},
			addend:   500,
			expected: []int32{1500, 1500, 1500, 1500},
		},
		{
			name:     "overflow wraps around",
			input:    []int32{2000000000, 2000000000, 2000000000, 2000000000},
			addend:   500000000,
			expected: []int32{-1794967296, -1794967296, -1794967296, -1794967296},
		},
		{
			name:     "underflow wraps around",
			input:    []int32{-2000000000, -2000000000, -2000000000, -2000000000},
			addend:   -500000000,
			expected: []int32{1794967296, 1794967296, 1794967296, 1794967296},
		},
		{
			name:     "add zero",
			input:    []int32{1000, 2000, 3000, 4000},
			addend:   0,
			expected: []int32{1000, 2000, 3000, 4000},
		},
		{
			name:     "mixed positive and negative",
			input:    []int32{-1000, 2000, -3000, 4000},
			addend:   500,
			expected: []int32{-500, 2500, -2500, 4500},
		},
		{
			name:     "all zeros",
			input:    []int32{0, 0, 0, 0},
			addend:   1000,
			expected: []int32{1000, 1000, 1000, 1000},
		},
		{
			name:     "boundary values",
			input:    []int32{2147483647, -2147483648, 2147483647, -2147483648},
			addend:   1,
			expected: []int32{-2147483648, -2147483647, -2147483648, -2147483647},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(AddInt32x4[int32](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []int32
		subtrahend int32
		expected   []int32
	}{
		{
			name:       "basic subtraction",
			input:      []int32{1000, 1000, 1000, 1000},
			subtrahend: 300,
			expected:   []int32{700, 700, 700, 700},
		},
		{
			name:       "underflow wraps around",
			input:      []int32{100, 100, 100, 100},
			subtrahend: 200,
			expected:   []int32{-100, -100, -100, -100},
		},
		{
			name:       "overflow wraps around (negative - negative)",
			input:      []int32{-2000000000, -2000000000, -2000000000, -2000000000},
			subtrahend: -500000000,
			expected:   []int32{-1500000000, -1500000000, -1500000000, -1500000000},
		},
		{
			name:       "subtract zero",
			input:      []int32{1000, 2000, 3000, 4000},
			subtrahend: 0,
			expected:   []int32{1000, 2000, 3000, 4000},
		},
		{
			name:       "mixed positive and negative",
			input:      []int32{-1000, 2000, -3000, 4000},
			subtrahend: 500,
			expected:   []int32{-1500, 1500, -3500, 3500},
		},
		{
			name:       "boundary values",
			input:      []int32{-2147483648, 2147483647, -2147483648, 2147483647},
			subtrahend: 1,
			expected:   []int32{2147483647, 2147483646, 2147483647, 2147483646},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(SubInt32x4[int32](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int32
		min       int32
		max       int32
		expected  []int32
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []int32{-1500, -500, 500, 2500},
			min:      -500,
			max:      1500,
			expected: []int32{-500, -500, 500, 1500},
		},
		{
			name:     "all below min",
			input:    []int32{-10000, -20000, -30000, -40000},
			min:      -5000,
			max:      5000,
			expected: []int32{-5000, -5000, -5000, -5000},
		},
		{
			name:     "all above max",
			input:    []int32{10000, 20000, 30000, 40000},
			min:      -5000,
			max:      5000,
			expected: []int32{5000, 5000, 5000, 5000},
		},
		{
			name:     "all in range",
			input:    []int32{-100, 0, 100, 500},
			min:      -1000,
			max:      1000,
			expected: []int32{-100, 0, 100, 500},
		},
		{
			name:     "min equals max",
			input:    []int32{-1000, 0, 1000, 5000},
			min:      50,
			max:      50,
			expected: []int32{50, 50, 50, 50},
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
					_ = ClampInt32x4[int32](tc.min, tc.max)(ro.Empty[*archsimd.Int32x4]())
				})
				return
			}

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(ClampInt32x4[int32](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int32
		threshold int32
		expected  []int32
	}{
		{
			name:      "basic min",
			input:     []int32{-1500, -500, 500, 2500},
			threshold: 0,
			expected:  []int32{0, 0, 500, 2500},
		},
		{
			name:      "all below threshold",
			input:     []int32{-10000, -20000, -30000, -40000},
			threshold: 0,
			expected:  []int32{0, 0, 0, 0},
		},
		{
			name:      "all above threshold",
			input:     []int32{1000, 2000, 3000, 4000},
			threshold: 0,
			expected:  []int32{1000, 2000, 3000, 4000},
		},
		{
			name:      "negative threshold",
			input:     []int32{-5000, -3000, -1000, 1000},
			threshold: -2000,
			expected:  []int32{-2000, -2000, -1000, 1000},
		},
		{
			name:      "boundary values",
			input:     []int32{2147483647, -2147483648, 0, 0},
			threshold: 0,
			expected:  []int32{2147483647, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(MinInt32x4[int32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []int32
		threshold int32
		expected  []int32
	}{
		{
			name:      "basic max",
			input:     []int32{-1500, -500, 500, 2500},
			threshold: 0,
			expected:  []int32{-1500, -500, 0, 0},
		},
		{
			name:      "all below threshold",
			input:     []int32{-10000, -20000, -30000, -40000},
			threshold: 0,
			expected:  []int32{-10000, -20000, -30000, -40000},
		},
		{
			name:      "all above threshold",
			input:     []int32{1000, 2000, 3000, 4000},
			threshold: 0,
			expected:  []int32{0, 0, 0, 0},
		},
		{
			name:      "positive threshold",
			input:     []int32{-5000, -3000, -1000, 1000},
			threshold: 500,
			expected:  []int32{-5000, -3000, -1000, 500},
		},
		{
			name:      "boundary values",
			input:     []int32{-2147483648, 2147483647, 0, 0},
			threshold: 0,
			expected:  []int32{-2147483648, 0, 0, 0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Int32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Int32x4ToScalar[int32]()(MaxInt32x4[int32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int32x4
		expected int32
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int32x4 {
				vec := archsimd.Int32x4{}
				for i := 0; i < 4; i++ {
					vec = vec.SetElem(uint8(i), int32(i*1000-1500))
				}
				return &vec
			},
			expected: -1500, // min is -1500 at index 0
		},
		{
			name: "all positive values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int32x4 {
				vec := archsimd.Int32x4{}
				for i := 0; i < 4; i++ {
					vec = vec.SetElem(uint8(i), int32(i+1)) // values 1-4
				}
				return &vec
			},
			expected: 1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int32x4 {
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

			var result []int32
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMinInt32x4[int32]()(ro.Empty[*archsimd.Int32x4]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMinInt32x4[int32]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

func TestReduceMaxInt32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		setupVec func() *archsimd.Int32x4
		expected int32
	}{
		{
			name: "mixed values",
			setupVec: func() *archsimd.Int32x4 {
				vec := archsimd.Int32x4{}
				for i := 0; i < 4; i++ {
					vec = vec.SetElem(uint8(i), int32(i*1000-1500))
				}
				return &vec
			},
			expected: 1500, // max is 1500 at index 3
		},
		{
			name: "all negative values - detects accumulator initialization bug",
			setupVec: func() *archsimd.Int32x4 {
				vec := archsimd.Int32x4{}
				for i := 0; i < 4; i++ {
					vec = vec.SetElem(uint8(i), int32(-int32(i)-1)) // values -1 to -4
				}
				return &vec
			},
			expected: -1, // before fix: would return 0 (from zero-initialized accumulator)
		},
		{
			name: "empty observable returns 0",
			setupVec: func() *archsimd.Int32x4 {
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

			var result []int32
			var err error

			if tc.setupVec() == nil {
				result, err = ro.Collect(
					ReduceMaxInt32x4[int32]()(ro.Empty[*archsimd.Int32x4]()),
				)
			} else {
				result, err = ro.Collect(
					ReduceMaxInt32x4[int32]()(ro.Just(tc.setupVec())),
				)
			}

			is.NoError(err)
			is.Equal([]int32{tc.expected}, result)
		})
	}
}

// Int64x2 tests

func TestReduceSumInt64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int64
		expected int64
	}{
		{
			name:     "basic sum",
			input:    []int64{1000000, 2000000},
			expected: 3000000,
		},
		{
			name:     "all zeros",
			input:    []int64{0, 0},
			expected: 0,
		},
		{
			name:     "mixed positive and negative",
			input:    []int64{-1000000, 2000000},
			expected: 1000000,
		},
		{
			name:     "all negative",
			input:    []int64{-1000000, -2000000},
			expected: -3000000,
		},
		{
			name:     "boundary values",
			input:    []int64{9223372036854775807, -9223372036854775808},
			expected: -1,
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
				ReduceSumInt64x2[int64]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]int64{tc.expected}, result)
		})
	}
}

func TestAddInt64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []int64
		addend   int64
		expected []int64
	}{
		{
			name:     "basic addition",
			input:    []int64{1000000, 1000000},
			addend:   500000,
			expected: []int64{1500000, 1500000},
		},
		{
			name:     "add zero",
			input:    []int64{1000000, 2000000},
			addend:   0,
			expected: []int64{1000000, 2000000},
		},
		{
			name:     "negative addend",
			input:    []int64{5000000, 3000000},
			addend:   -1000000,
			expected: []int64{4000000, 2000000},
		},
		{
			name:     "mixed signs",
			input:    []int64{-1000000, 1000000},
			addend:   500000,
			expected: []int64{-500000, 1500000},
		},
		{
			name:     "boundary overflow",
			input:    []int64{9223372036854775807, 9223372036854775807},
			addend:   1,
			expected: []int64{-9223372036854775808, -9223372036854775808},
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
				Int64x2ToScalar[int64]()(AddInt64x2[int64](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubInt64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []int64
		subtrahend int64
		expected   []int64
	}{
		{
			name:       "basic subtraction",
			input:      []int64{1000000, 1000000},
			subtrahend: 300000,
			expected:   []int64{700000, 700000},
		},
		{
			name:       "subtract zero",
			input:      []int64{1000000, 2000000},
			subtrahend: 0,
			expected:   []int64{1000000, 2000000},
		},
		{
			name:       "negative subtrahend",
			input:      []int64{1000000, 2000000},
			subtrahend: -500000,
			expected:   []int64{1500000, 2500000},
		},
		{
			name:       "underflow wraps",
			input:      []int64{100, 100},
			subtrahend: 200,
			expected:   []int64{-100, -100},
		},
		{
			name:       "boundary values",
			input:      []int64{-9223372036854775808, 9223372036854775807},
			subtrahend: 1,
			expected:   []int64{9223372036854775807, 9223372036854775806},
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
				Int64x2ToScalar[int64]()(SubInt64x2[int64](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// Uint8x16 tests

func TestReduceSumUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name:     "sequential 1-16",
			input:    []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			expected: 136,
		},
		{
			name:     "all zeros",
			input:    []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "all ones",
			input:    []uint8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			expected: 16,
		},
		{
			name: "overflow wraps",
			input: []uint8{20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20},
			expected: 64, // 320 % 256
		},
		{
			name:     "boundary",
			input:    []uint8{255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0},
			expected: 248, // 8*255 % 256
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumUint8x16[uint8]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

func TestAddUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint8
		addend   uint8
		expected []uint8
	}{
		{
			name:     "basic addition",
			input:    []uint8{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
			addend:   30,
			expected: []uint8{80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80},
		},
		{
			name:     "overflow wraps around",
			input:    []uint8{200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200},
			addend:   100,
			expected: []uint8{44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44},
		},
		{
			name:     "add zero",
			input:    []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
			addend:   0,
			expected: []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
		},
		{
			name:     "all zeros",
			input:    []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			addend:   100,
			expected: []uint8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
		},
		{
			name:     "boundary values",
			input:    []uint8{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0},
			addend:   1,
			expected: []uint8{0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
		},
		{
			name:     "sequence",
			input:    []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			addend:   10,
			expected: []uint8{11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(AddUint8x16[uint8](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []uint8
		subtrahend uint8
		expected   []uint8
	}{
		{
			name:       "basic subtraction",
			input:      []uint8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
			subtrahend: 30,
			expected:   []uint8{70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70},
		},
		{
			name:       "underflow wraps around",
			input:      []uint8{10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10},
			subtrahend: 20,
			expected:   []uint8{246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246, 246},
		},
		{
			name:       "subtract zero",
			input:      []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
			subtrahend: 0,
			expected:   []uint8{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160},
		},
		{
			name:       "all zeros",
			input:      []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
			subtrahend: 0,
			expected:   []uint8{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		},
		{
			name:       "boundary values",
			input:      []uint8{255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0},
			subtrahend: 1,
			expected:   []uint8{254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255, 254, 255},
		},
		{
			name:       "sequence",
			input:      []uint8{100, 90, 80, 70, 60, 50, 40, 30, 20, 10, 9, 8, 7, 6, 5, 4},
			subtrahend: 5,
			expected:   []uint8{95, 85, 75, 65, 55, 45, 35, 25, 15, 5, 4, 3, 2, 1, 0, 255},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(SubUint8x16[uint8](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint8
		min       uint8
		max       uint8
		expected  []uint8
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []uint8{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
			min:      20,
			max:      100,
			expected: []uint8{20, 20, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100, 100, 100, 100, 100},
		},
		{
			name:     "all below min",
			input:    []uint8{0, 5, 10, 15, 0, 5, 10, 15, 0, 5, 10, 15, 0, 5, 10, 15},
			min:      50,
			max:      100,
			expected: []uint8{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
		},
		{
			name:     "all above max",
			input:    []uint8{150, 200, 250, 255, 150, 200, 250, 255, 150, 200, 250, 255, 150, 200, 250, 255},
			min:      50,
			max:      100,
			expected: []uint8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
		},
		{
			name:     "all in range",
			input:    []uint8{60, 65, 70, 75, 80, 85, 90, 95, 60, 65, 70, 75, 80, 85, 90, 95},
			min:      50,
			max:      100,
			expected: []uint8{60, 65, 70, 75, 80, 85, 90, 95, 60, 65, 70, 75, 80, 85, 90, 95},
		},
		{
			name:     "min equals max",
			input:    []uint8{0, 50, 100, 150, 200, 250, 0, 50, 100, 150, 200, 250, 0, 50, 100, 150},
			min:      100,
			max:      100,
			expected: []uint8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100},
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
					_ = ClampUint8x16[uint8](tc.min, tc.max)(ro.Empty[*archsimd.Uint8x16]())
				})
				return
			}

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(ClampUint8x16[uint8](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name:      "basic min - keeps larger of value and threshold",
			input:     []uint8{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
			threshold: 50,
			expected:  []uint8{50, 50, 50, 50, 50, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
		},
		{
			name:      "all below threshold",
			input:     []uint8{0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0},
			threshold: 50,
			expected:  []uint8{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
		},
		{
			name:      "all above threshold",
			input:     []uint8{100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250},
			threshold: 50,
			expected:  []uint8{100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250},
		},
		{
			name:      "threshold zero",
			input:     []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			threshold: 0,
			expected:  []uint8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(MinUint8x16[uint8](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint8
		threshold uint8
		expected  []uint8
	}{
		{
			name:      "basic max - keeps smaller of value and threshold",
			input:     []uint8{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
			threshold: 50,
			expected:  []uint8{0, 10, 20, 30, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
		},
		{
			name:      "all below threshold",
			input:     []uint8{0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0},
			threshold: 50,
			expected:  []uint8{0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0, 10, 20, 30, 40, 0},
		},
		{
			name:      "all above threshold",
			input:     []uint8{100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250},
			threshold: 50,
			expected:  []uint8{50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50},
		},
		{
			name:      "threshold 255",
			input:     []uint8{0, 50, 100, 150, 200, 250, 0, 50, 100, 150, 200, 250, 0, 50, 100, 150},
			threshold: 255,
			expected:  []uint8{0, 50, 100, 150, 200, 250, 0, 50, 100, 150, 200, 250, 0, 50, 100, 150},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint8x16ToScalar[uint8]()(MaxUint8x16[uint8](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name:     "sequential values",
			input:    []uint8{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
			expected: 0,
		},
		{
			name: "all same",
			input: []uint8{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
			expected: 42,
		},
		{
			name:     "min at end",
			input:    []uint8{100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1},
			expected: 1,
		},
		{
			name:     "boundary zero",
			input:    []uint8{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMinUint8x16[uint8]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint8x16(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint8
		expected uint8
	}{
		{
			name:     "sequential values - max 150 at index 15",
			input:    []uint8{0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150},
			expected: 150,
		},
		{
			name: "all same",
			input: []uint8{42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42},
			expected: 42,
		},
		{
			name:     "max at start",
			input:    []uint8{255, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
			expected: 255,
		},
		{
			name:     "all max",
			input:    []uint8{255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255},
			expected: 255,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint8x16{}
			for i := 0; i < 16; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMaxUint8x16[uint8]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint8{tc.expected}, result)
		})
	}
}

// Uint16x8 tests

func TestReduceSumUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name:     "basic sum",
			input:    []uint16{100, 200, 300, 400, 500, 600, 700, 800},
			expected: 3600,
		},
		{
			name:     "all zeros",
			input:    []uint16{0, 0, 0, 0, 0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "all ones",
			input:    []uint16{1, 1, 1, 1, 1, 1, 1, 1},
			expected: 8,
		},
		{
			name:     "overflow wraps",
			input:    []uint16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 10000},
			expected: 14464, // 80000 % 65536
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumUint16x8[uint16]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

func TestAddUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint16
		addend   uint16
		expected []uint16
	}{
		{
			name:     "basic addition",
			input:    []uint16{100, 100, 100, 100, 100, 100, 100, 100},
			addend:   50,
			expected: []uint16{150, 150, 150, 150, 150, 150, 150, 150},
		},
		{
			name:     "add zero",
			input:    []uint16{100, 200, 300, 400, 500, 600, 700, 800},
			addend:   0,
			expected: []uint16{100, 200, 300, 400, 500, 600, 700, 800},
		},
		{
			name:     "overflow wraps",
			input:    []uint16{60000, 60000, 60000, 60000, 60000, 60000, 60000, 60000},
			addend:   10000,
			expected: []uint16{4464, 4464, 4464, 4464, 4464, 4464, 4464, 4464},
		},
		{
			name:     "boundary",
			input:    []uint16{65535, 0, 65535, 0, 65535, 0, 65535, 0},
			addend:   1,
			expected: []uint16{0, 1, 0, 1, 0, 1, 0, 1},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(AddUint16x8[uint16](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []uint16
		subtrahend uint16
		expected   []uint16
	}{
		{
			name:       "basic subtraction",
			input:      []uint16{500, 500, 500, 500, 500, 500, 500, 500},
			subtrahend: 100,
			expected:   []uint16{400, 400, 400, 400, 400, 400, 400, 400},
		},
		{
			name:       "subtract zero",
			input:      []uint16{100, 200, 300, 400, 500, 600, 700, 800},
			subtrahend: 0,
			expected:   []uint16{100, 200, 300, 400, 500, 600, 700, 800},
		},
		{
			name:       "underflow wraps",
			input:      []uint16{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000},
			subtrahend: 2000,
			expected:   []uint16{64536, 64536, 64536, 64536, 64536, 64536, 64536, 64536},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(SubUint16x8[uint16](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint16
		min       uint16
		max       uint16
		expected  []uint16
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 7000},
			min:      1000,
			max:      5000,
			expected: []uint16{1000, 1000, 2000, 3000, 4000, 5000, 5000, 5000},
		},
		{
			name:     "all below min",
			input:    []uint16{0, 100, 200, 300, 400, 500, 600, 700},
			min:      1000,
			max:      5000,
			expected: []uint16{1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000},
		},
		{
			name:     "all above max",
			input:    []uint16{10000, 20000, 30000, 40000, 50000, 60000, 10000, 20000},
			min:      1000,
			max:      5000,
			expected: []uint16{5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
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
					_ = ClampUint16x8[uint16](tc.min, tc.max)(ro.Empty[*archsimd.Uint16x8]())
				})
				return
			}

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(ClampUint16x8[uint16](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint16
		threshold uint16
		expected  []uint16
	}{
		{
			name:      "basic min",
			input:     []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 7000},
			threshold: 3000,
			expected:  []uint16{3000, 3000, 3000, 3000, 4000, 5000, 6000, 7000},
		},
		{
			name:      "all below threshold",
			input:     []uint16{0, 500, 1000, 1500, 2000, 2500, 0, 500},
			threshold: 3000,
			expected:  []uint16{3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000},
		},
		{
			name:      "all above threshold",
			input:     []uint16{5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000},
			threshold: 3000,
			expected:  []uint16{5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(MinUint16x8[uint16](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint16
		threshold uint16
		expected  []uint16
	}{
		{
			name:      "basic max",
			input:     []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 7000},
			threshold: 3000,
			expected:  []uint16{0, 1000, 2000, 3000, 3000, 3000, 3000, 3000},
		},
		{
			name:      "all below threshold",
			input:     []uint16{0, 500, 1000, 1500, 2000, 2500, 0, 500},
			threshold: 3000,
			expected:  []uint16{0, 500, 1000, 1500, 2000, 2500, 0, 500},
		},
		{
			name:      "all above threshold",
			input:     []uint16{5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000},
			threshold: 3000,
			expected:  []uint16{3000, 3000, 3000, 3000, 3000, 3000, 3000, 3000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint16x8ToScalar[uint16]()(MaxUint16x8[uint16](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name:     "sequential values",
			input:    []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 7000},
			expected: 0,
		},
		{
			name:     "all same",
			input:    []uint16{5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
			expected: 5000,
		},
		{
			name:     "min at end",
			input:    []uint16{10000, 10000, 10000, 10000, 10000, 10000, 10000, 1},
			expected: 1,
		},
		{
			name:     "boundary",
			input:    []uint16{65535, 1000, 2000, 3000, 4000, 5000, 6000, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMinUint16x8[uint16]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint16x8(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint16
		expected uint16
	}{
		{
			name:     "sequential values",
			input:    []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 7000},
			expected: 7000,
		},
		{
			name:     "all same",
			input:    []uint16{5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000},
			expected: 5000,
		},
		{
			name:     "max at start",
			input:    []uint16{65535, 1, 2, 3, 4, 5, 6, 7},
			expected: 65535,
		},
		{
			name:     "boundary",
			input:    []uint16{0, 1000, 2000, 3000, 4000, 5000, 6000, 65535},
			expected: 65535,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint16x8{}
			for i := 0; i < 8; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMaxUint16x8[uint16]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint16{tc.expected}, result)
		})
	}
}

// Uint32x4 tests

func TestReduceSumUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name:     "basic sum",
			input:    []uint32{1000, 2000, 3000, 4000},
			expected: 10000,
		},
		{
			name:     "all zeros",
			input:    []uint32{0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "overflow wraps",
			input:    []uint32{2000000000, 2000000000, 2000000000, 2000000000},
			expected: 3705032704, // 8e9 % 2^32
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumUint32x4[uint32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

func TestAddUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint32
		addend   uint32
		expected []uint32
	}{
		{
			name:     "basic addition",
			input:    []uint32{1000, 1000, 1000, 1000},
			addend:   500,
			expected: []uint32{1500, 1500, 1500, 1500},
		},
		{
			name:     "add zero",
			input:    []uint32{1000, 2000, 3000, 4000},
			addend:   0,
			expected: []uint32{1000, 2000, 3000, 4000},
		},
		{
			name:     "overflow wraps",
			input:    []uint32{4000000000, 4000000000, 4000000000, 4000000000},
			addend:   500000000,
			expected: []uint32{205032704, 205032704, 205032704, 205032704}, // 4.5e9 % 2^32
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(AddUint32x4[uint32](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []uint32
		subtrahend uint32
		expected   []uint32
	}{
		{
			name:       "basic subtraction",
			input:      []uint32{5000, 5000, 5000, 5000},
			subtrahend: 200,
			expected:   []uint32{4800, 4800, 4800, 4800},
		},
		{
			name:       "subtract zero",
			input:      []uint32{1000, 2000, 3000, 4000},
			subtrahend: 0,
			expected:   []uint32{1000, 2000, 3000, 4000},
		},
		{
			name:       "underflow wraps",
			input:      []uint32{100, 100, 100, 100},
			subtrahend: 500,
			expected:   []uint32{4294966896, 4294966896, 4294966896, 4294966896},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(SubUint32x4[uint32](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestClampUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint32
		min       uint32
		max       uint32
		expected  []uint32
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []uint32{0, 5000, 10000, 20000},
			min:      5000,
			max:      15000,
			expected: []uint32{5000, 5000, 10000, 15000},
		},
		{
			name:     "all below min",
			input:    []uint32{0, 1000, 2000, 3000},
			min:      5000,
			max:      15000,
			expected: []uint32{5000, 5000, 5000, 5000},
		},
		{
			name:     "all above max",
			input:    []uint32{20000, 30000, 4000000000, 5000},
			min:      5000,
			max:      15000,
			expected: []uint32{15000, 15000, 15000, 5000},
		},
		{
			name:      "panic on min > max",
			input:     []uint32{},
			min:       50000,
			max:       15000,
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
					_ = ClampUint32x4[uint32](tc.min, tc.max)(ro.Empty[*archsimd.Uint32x4]())
				})
				return
			}

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(ClampUint32x4[uint32](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMinUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint32
		threshold uint32
		expected  []uint32
	}{
		{
			name:      "basic min",
			input:     []uint32{0, 5000, 10000, 15000},
			threshold: 10000,
			expected:  []uint32{10000, 10000, 10000, 15000},
		},
		{
			name:      "all below threshold",
			input:     []uint32{0, 1000, 2000, 3000},
			threshold: 10000,
			expected:  []uint32{10000, 10000, 10000, 10000},
		},
		{
			name:      "all above threshold",
			input:     []uint32{20000, 30000, 40000, 50000},
			threshold: 10000,
			expected:  []uint32{20000, 30000, 40000, 50000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(MinUint32x4[uint32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestMaxUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []uint32
		threshold uint32
		expected  []uint32
	}{
		{
			name:      "basic max",
			input:     []uint32{0, 5000, 10000, 15000},
			threshold: 10000,
			expected:  []uint32{0, 5000, 10000, 10000},
		},
		{
			name:      "all below threshold",
			input:     []uint32{0, 1000, 2000, 3000},
			threshold: 10000,
			expected:  []uint32{0, 1000, 2000, 3000},
		},
		{
			name:      "all above threshold",
			input:     []uint32{20000, 30000, 40000, 50000},
			threshold: 10000,
			expected:  []uint32{10000, 10000, 10000, 10000},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Uint32x4ToScalar[uint32]()(MaxUint32x4[uint32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestReduceMinUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name:     "sequential values",
			input:    []uint32{0, 5000, 10000, 15000},
			expected: 0,
		},
		{
			name:     "all same",
			input:    []uint32{50000, 50000, 50000, 50000},
			expected: 50000,
		},
		{
			name:     "min at end",
			input:    []uint32{100000, 200000, 300000, 1},
			expected: 1,
		},
		{
			name:     "boundary",
			input:    []uint32{4294967295, 1000, 2000, 0},
			expected: 0,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMinUint32x4[uint32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

func TestReduceMaxUint32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint32
		expected uint32
	}{
		{
			name:     "sequential values",
			input:    []uint32{0, 5000, 10000, 15000},
			expected: 15000,
		},
		{
			name:     "all same",
			input:    []uint32{50000, 50000, 50000, 50000},
			expected: 50000,
		},
		{
			name:     "max at start",
			input:    []uint32{4294967295, 1, 2, 3},
			expected: 4294967295,
		},
		{
			name:     "boundary",
			input:    []uint32{0, 1000, 2000, 4294967295},
			expected: 4294967295,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Uint32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMaxUint32x4[uint32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint32{tc.expected}, result)
		})
	}
}

// Uint64x2 tests

func TestReduceSumUint64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint64
		expected uint64
	}{
		{
			name:     "basic sum",
			input:    []uint64{1000000, 2000000},
			expected: 3000000,
		},
		{
			name:     "all zeros",
			input:    []uint64{0, 0},
			expected: 0,
		},
		{
			name:     "overflow wraps",
			input:    []uint64{9223372036854775807, 9223372036854775807},
			expected: 18446744073709551614,
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
				ReduceSumUint64x2[uint64]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Equal([]uint64{tc.expected}, result)
		})
	}
}

func TestAddUint64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []uint64
		addend   uint64
		expected []uint64
	}{
		{
			name:     "basic addition",
			input:    []uint64{1000000, 1000000},
			addend:   500000,
			expected: []uint64{1500000, 1500000},
		},
		{
			name:     "add zero",
			input:    []uint64{1000000, 2000000},
			addend:   0,
			expected: []uint64{1000000, 2000000},
		},
		{
			name:     "overflow wraps",
			input:    []uint64{9223372036854775807, 9223372036854775807},
			addend:   1,
			expected: []uint64{9223372036854775808, 9223372036854775808},
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
				Uint64x2ToScalar[uint64]()(AddUint64x2[uint64](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

func TestSubUint64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []uint64
		subtrahend uint64
		expected   []uint64
	}{
		{
			name:       "basic subtraction",
			input:      []uint64{5000000, 5000000},
			subtrahend: 2000000,
			expected:   []uint64{3000000, 3000000},
		},
		{
			name:       "subtract zero",
			input:      []uint64{1000000, 2000000},
			subtrahend: 0,
			expected:   []uint64{1000000, 2000000},
		},
		{
			name:       "underflow wraps",
			input:      []uint64{100, 100},
			subtrahend: 200,
			expected:   []uint64{18446744073709551516, 18446744073709551516},
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
				Uint64x2ToScalar[uint64]()(SubUint64x2[uint64](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			is.Equal(tc.expected, result)
		})
	}
}

// Float32x4 tests

func TestReduceSumFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name:     "basic sum",
			input:    []float32{1.5, 2.5, 3.5, 4.5},
			expected: 12,
		},
		{
			name:     "all zeros",
			input:    []float32{0, 0, 0, 0},
			expected: 0,
		},
		{
			name:     "negative values",
			input:    []float32{-1.0, -2.0, -3.0, -4.0},
			expected: -10,
		},
		{
			name:     "mixed positive and negative",
			input:    []float32{10.0, -5.0, 3.0, -2.0},
			expected: 6,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4},
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

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumFloat32x4[float32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Len(result, 1)
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

func TestAddFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float32
		addend   float32
		expected []float32
	}{
		{
			name:     "basic addition",
			input:    []float32{1.5, 1.5, 1.5, 1.5},
			addend:   2.5,
			expected: []float32{4.0, 4.0, 4.0, 4.0},
		},
		{
			name:     "add zero",
			input:    []float32{1.0, 2.0, 3.0, 4.0},
			addend:   0,
			expected: []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:     "negative addend",
			input:    []float32{10.0, 20.0, 30.0, 40.0},
			addend:   -5.0,
			expected: []float32{5.0, 15.0, 25.0, 35.0},
		},
		{
			name:     "mixed signs",
			input:    []float32{-5.0, 5.0, -10.0, 10.0},
			addend:   3.0,
			expected: []float32{-2.0, 8.0, -7.0, 13.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(AddFloat32x4[float32](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []float32
		subtrahend float32
		expected   []float32
	}{
		{
			name:       "basic subtraction",
			input:      []float32{5.5, 5.5, 5.5, 5.5},
			subtrahend: 2.5,
			expected:   []float32{3.0, 3.0, 3.0, 3.0},
		},
		{
			name:       "subtract zero",
			input:      []float32{1.0, 2.0, 3.0, 4.0},
			subtrahend: 0,
			expected:   []float32{1.0, 2.0, 3.0, 4.0},
		},
		{
			name:       "negative result",
			input:      []float32{1.0, 2.0, 3.0, 4.0},
			subtrahend: 10.0,
			expected:   []float32{-9.0, -8.0, -7.0, -6.0},
		},
		{
			name:       "negative subtrahend",
			input:      []float32{5.0, 10.0, 15.0, 20.0},
			subtrahend: -5.0,
			expected:   []float32{10.0, 15.0, 20.0, 25.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(SubFloat32x4[float32](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []float32
		min       float32
		max       float32
		expected  []float32
		wantPanic bool
	}{
		{
			name:     "basic clamp",
			input:    []float32{0, 5, 10, 15},
			min:      5,
			max:      10,
			expected: []float32{5, 5, 10, 10},
		},
		{
			name:     "all below min",
			input:    []float32{-10, -5, 0, 2},
			min:      5,
			max:      10,
			expected: []float32{5, 5, 5, 5},
		},
		{
			name:     "all above max",
			input:    []float32{15, 20, 100, 1000},
			min:      5,
			max:      10,
			expected: []float32{10, 10, 10, 10},
		},
		{
			name:     "all in range",
			input:    []float32{6, 7, 8, 9},
			min:      5,
			max:      10,
			expected: []float32{6, 7, 8, 9},
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
					_ = ClampFloat32x4[float32](tc.min, tc.max)(ro.Empty[*archsimd.Float32x4]())
				})
				return
			}

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(ClampFloat32x4[float32](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []float32
		threshold float32
		expected  []float32
	}{
		{
			name:      "basic min",
			input:     []float32{0, 5, 10, 15},
			threshold: 5,
			expected:  []float32{5, 5, 10, 15},
		},
		{
			name:      "all below threshold",
			input:     []float32{-10, -5, 0, 2},
			threshold: 5,
			expected:  []float32{5, 5, 5, 5},
		},
		{
			name:      "all above threshold",
			input:     []float32{10, 15, 20, 25},
			threshold: 5,
			expected:  []float32{10, 15, 20, 25},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(MinFloat32x4[float32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []float32
		threshold float32
		expected  []float32
	}{
		{
			name:      "basic max",
			input:     []float32{0, 5, 10, 15},
			threshold: 5,
			expected:  []float32{0, 5, 5, 5},
		},
		{
			name:      "all below threshold",
			input:     []float32{-10, -5, 0, 2},
			threshold: 5,
			expected:  []float32{-10, -5, 0, 2},
		},
		{
			name:      "all above threshold",
			input:     []float32{10, 15, 20, 25},
			threshold: 5,
			expected:  []float32{5, 5, 5, 5},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float32x4ToScalar[float32]()(MaxFloat32x4[float32](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceMinFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name:     "sequential values",
			input:    []float32{0, 5, 10, 15},
			expected: 0,
		},
		{
			name:     "all same",
			input:    []float32{3.14, 3.14, 3.14, 3.14},
			expected: 3.14,
		},
		{
			name:     "negative values",
			input:    []float32{-10, -5, -20, -1},
			expected: -20,
		},
		{
			name:     "min at end",
			input:    []float32{100, 50, 25, 0.1},
			expected: 0.1,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4},
			expected: float32(math.NaN()),
		},
		{
			name:     "negative Inf is min",
			input:    []float32{1, 2, float32(math.Inf(-1)), 4},
			expected: float32(math.Inf(-1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMinFloat32x4[float32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Len(result, 1)
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

func TestReduceMaxFloat32x4(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float32
		expected float32
	}{
		{
			name:     "sequential values",
			input:    []float32{0, 5, 10, 15},
			expected: 15,
		},
		{
			name:     "all same",
			input:    []float32{3.14, 3.14, 3.14, 3.14},
			expected: 3.14,
		},
		{
			name:     "negative values",
			input:    []float32{-10, -5, -20, -1},
			expected: -1,
		},
		{
			name:     "max at start",
			input:    []float32{100, 50, 25, 0.1},
			expected: 100,
		},
		{
			name:     "NaN propagates",
			input:    []float32{1, 2, float32(math.NaN()), 4},
			expected: float32(math.NaN()),
		},
		{
			name:     "positive Inf is max",
			input:    []float32{1, 2, float32(math.Inf(1)), 4},
			expected: float32(math.Inf(1)),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float32x4{}
			for i := 0; i < 4; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMaxFloat32x4[float32]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Len(result, 1)
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

// Float64x2 tests

func TestReduceSumFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "basic sum",
			input:    []float64{1.5, 2.5},
			expected: 4,
		},
		{
			name:     "all zeros",
			input:    []float64{0, 0},
			expected: 0,
		},
		{
			name:     "negative values",
			input:    []float64{-10.5, -5.5},
			expected: -16,
		},
		{
			name:     "mixed signs",
			input:    []float64{100.0, -50.0},
			expected: 50,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, math.NaN()},
			expected: math.NaN(),
		},
		{
			name:     "positive Inf",
			input:    []float64{1, math.Inf(1)},
			expected: math.Inf(1),
		},
		{
			name:     "negative Inf",
			input:    []float64{math.Inf(-1), 2},
			expected: math.Inf(-1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceSumFloat64x2[float64]()(ro.Just(&vec)),
			)

			is.NoError(err)
			is.Len(result, 1)
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

func TestAddFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float64
		addend   float64
		expected []float64
	}{
		{
			name:     "basic addition",
			input:    []float64{1.5, 1.5},
			addend:   2.5,
			expected: []float64{4.0, 4.0},
		},
		{
			name:     "add zero",
			input:    []float64{1.0, 2.0},
			addend:   0,
			expected: []float64{1.0, 2.0},
		},
		{
			name:     "negative addend",
			input:    []float64{10.0, 20.0},
			addend:   -5.0,
			expected: []float64{5.0, 15.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(AddFloat64x2[float64](tc.addend)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestSubFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name       string
		input      []float64
		subtrahend float64
		expected   []float64
	}{
		{
			name:       "basic subtraction",
			input:      []float64{5.5, 5.5},
			subtrahend: 2.5,
			expected:   []float64{3.0, 3.0},
		},
		{
			name:       "subtract zero",
			input:      []float64{1.0, 2.0},
			subtrahend: 0,
			expected:   []float64{1.0, 2.0},
		},
		{
			name:       "negative result",
			input:      []float64{1.0, 2.0},
			subtrahend: 10.0,
			expected:   []float64{-9.0, -8.0},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(SubFloat64x2[float64](tc.subtrahend)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestClampFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

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
			input:    []float64{0, 10},
			min:      5,
			max:      10,
			expected: []float64{5, 10},
		},
		{
			name:     "all below min",
			input:    []float64{-10, 0},
			min:      5,
			max:      10,
			expected: []float64{5, 5},
		},
		{
			name:     "all above max",
			input:    []float64{100, 200},
			min:      5,
			max:      10,
			expected: []float64{10, 10},
		},
		{
			name:      "panic on min > max",
			input:     []float64{},
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
					_ = ClampFloat64x2[float64](tc.min, tc.max)(ro.Empty[*archsimd.Float64x2]())
				})
				return
			}

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(ClampFloat64x2[float64](tc.min, tc.max)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMinFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic min",
			input:     []float64{0, 10},
			threshold: 5,
			expected:  []float64{5, 10},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10, 0},
			threshold: 5,
			expected:  []float64{5, 5},
		},
		{
			name:      "all above threshold",
			input:     []float64{10, 20},
			threshold: 5,
			expected:  []float64{10, 20},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(MinFloat64x2[float64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestMaxFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name      string
		input     []float64
		threshold float64
		expected  []float64
	}{
		{
			name:      "basic max",
			input:     []float64{0, 10},
			threshold: 5,
			expected:  []float64{0, 5},
		},
		{
			name:      "all below threshold",
			input:     []float64{-10, 0},
			threshold: 5,
			expected:  []float64{-10, 0},
		},
		{
			name:      "all above threshold",
			input:     []float64{10, 20},
			threshold: 5,
			expected:  []float64{5, 5},
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				Float64x2ToScalar[float64]()(MaxFloat64x2[float64](tc.threshold)(ro.Just(&vec))),
			)

			is.NoError(err)
			for i, v := range result {
				is.InDelta(tc.expected[i], v, 0.0001)
			}
		})
	}
}

func TestReduceMinFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "sequential values",
			input:    []float64{0, 5},
			expected: 0,
		},
		{
			name:     "all same",
			input:    []float64{3.14, 3.14},
			expected: 3.14,
		},
		{
			name:     "negative values",
			input:    []float64{-10.0, -5.0},
			expected: -10,
		},
		{
			name:     "min at end",
			input:    []float64{100, 0.1},
			expected: 0.1,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, math.NaN()},
			expected: math.NaN(),
		},
		{
			name:     "negative Inf is min",
			input:    []float64{1, math.Inf(-1)},
			expected: math.Inf(-1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMinFloat64x2[float64]()(ro.Just(&vec)),
			)

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

func TestReduceMaxFloat64x2(t *testing.T) {
	t.Parallel()
	requireAVX(t)

	testCases := []struct {
		name     string
		input    []float64
		expected float64
	}{
		{
			name:     "sequential values",
			input:    []float64{0, 5},
			expected: 5,
		},
		{
			name:     "all same",
			input:    []float64{3.14, 3.14},
			expected: 3.14,
		},
		{
			name:     "negative values",
			input:    []float64{-10.0, -5.0},
			expected: -5,
		},
		{
			name:     "max at start",
			input:    []float64{100, 0.1},
			expected: 100,
		},
		{
			name:     "NaN propagates",
			input:    []float64{1, math.NaN()},
			expected: math.NaN(),
		},
		{
			name:     "positive Inf is max",
			input:    []float64{1, math.Inf(1)},
			expected: math.Inf(1),
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			vec := archsimd.Float64x2{}
			for i := 0; i < 2; i++ {
				vec = vec.SetElem(uint8(i), tc.input[i])
			}

			result, err := ro.Collect(
				ReduceMaxFloat64x2[float64]()(ro.Just(&vec)),
			)

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
