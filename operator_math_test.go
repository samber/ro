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

package ro

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestOperatorMathAverage(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Average[int]()(Just(1, 2, 3)),
	)
	is.Equal([]float64{2}, values)
	is.NoError(err)

	values, err = Collect(
		Average[int]()(Just(1, 2)),
	)
	is.Equal([]float64{1.5}, values)
	is.NoError(err)

	values, err = Collect(
		Average[int]()(Just(1, -1)),
	)
	is.Equal([]float64{0}, values)
	is.NoError(err)

	values, err = Collect(
		Average[int]()(Empty[int]()),
	)
	is.True(math.IsNaN(values[0]))
	is.NoError(err)

	values, err = Collect(
		Average[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]float64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathCount(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Count[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int64{3}, values)
	is.NoError(err)

	values, err = Collect(
		Count[int]()(Empty[int]()),
	)
	is.Equal([]int64{0}, values)
	is.NoError(err)

	values, err = Collect(
		Count[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathSum(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Sum[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{6}, values)
	is.NoError(err)

	values, err = Collect(
		Sum[int]()(Empty[int]()),
	)
	is.Equal([]int{0}, values)
	is.NoError(err)

	values, err = Collect(
		Sum[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathRound(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathMin(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Min[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Min[int]()(Just(3, 2, 1, -42)),
	)
	is.Equal([]int{-42}, values)
	is.NoError(err)

	values, err = Collect(
		Min[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Min[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathMax(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Max[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{3}, values)
	is.NoError(err)

	values, err = Collect(
		Max[int]()(Just(3, 2, 1, -42)),
	)
	is.Equal([]int{3}, values)
	is.NoError(err)

	values, err = Collect(
		Max[int]()(Empty[int]()),
	)
	is.Equal([]int{0}, values)
	is.NoError(err)

	values, err = Collect(
		Max[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathClamp(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathAbs(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathFloor(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathFloorWithPrecision(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name      string
		precision int
		source    Observable[float64]
		want      []float64
		wantErr   error
	}{
		{
			name:      "positive precision",
			precision: 2,
			source:    Just(3.14159, 2.71828, 0.0),
			want:      []float64{3.14, 2.71, 0},
		},
		{
			name:      "negative values",
			precision: 2,
			source:    Just(-1.2345, -9.8765),
			want:      []float64{-1.24, -9.88},
		},
		{
			name:      "zero precision",
			precision: 0,
			source:    Just(2.3, -2.3, 5.0),
			want:      []float64{2, -3, 5},
		},
		{
			name:      "very small numbers",
			precision: 4,
			source:    Just(0.000123, -0.000987),
			want:      []float64{0.0001, -0.001},
		},
		{
			name:      "error propagation",
			precision: 3,
			source:    Throw[float64](assert.AnError),
			want:      []float64{},
			wantErr:   assert.AnError,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()

			is := assert.New(t)

			values, err := Collect(
				FloorWithPrecision(tt.precision)(tt.source),
			)

			if tt.wantErr != nil {
				is.Equal(tt.want, values)
				is.EqualError(err, tt.wantErr.Error())
				return
			}

			is.NoError(err)
			is.Equal(len(tt.want), len(values))
			for i := range tt.want {
				is.InDelta(tt.want[i], values[i], 1e-9)
			}
		})
	}

	t.Run("negative precision panics", func(t *testing.T) {
		t.Parallel()
		assert.Panics(t, func() {
			FloorWithPrecision(-1)
		})
	})
}

func TestOperatorMathCeil(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathTrunc(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorMathReduce(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reducer := func(agg, current int) int {
		return agg + current
	}

	values, err := Collect(
		Reduce(reducer, 10)(Just(1, 2, 3)),
	)
	is.Equal([]int{16}, values)
	is.NoError(err)

	values, err = Collect(
		Reduce(reducer, 10)(Empty[int]()),
	)
	is.Equal([]int{10}, values)
	is.NoError(err)

	values, err = Collect(
		Reduce(reducer, 10)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorMathReduceI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reducer := func(agg, current int, _ int64) int {
		return agg + current
	}

	values, err := Collect(
		ReduceI(func(agg, current int, i int64) int {
			is.Equal(current, int(i))
			return agg + current
		}, 10)(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{16}, values)
	is.NoError(err)

	values, err = Collect(
		ReduceI(reducer, 10)(Just(1, 2, 3)),
	)
	is.Equal([]int{16}, values)
	is.NoError(err)

	values, err = Collect(
		ReduceI(reducer, 10)(Empty[int]()),
	)
	is.Equal([]int{10}, values)
	is.NoError(err)

	values, err = Collect(
		ReduceI(reducer, 10)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}
