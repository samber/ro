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

func TestOperatorMathCeil(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Ceil()(Just(3.2, 4.7, -2.3, -5.8, 0.0, 7.0)),
	)
	is.Equal([]float64{4, 5, -2, -5, 0, 7}, values)
	is.NoError(err)

	values, err = Collect(
		Ceil()(Just(math.Inf(-1), -42.7, math.Inf(1))),
	)
	is.NoError(err)
	is.Len(values, 3)
	is.True(math.IsInf(values[0], -1))
	is.Equal(float64(-42), values[1])
	is.True(math.IsInf(values[2], 1))

	values, err = Collect(
		Ceil()(Just(math.NaN())),
	)
	is.NoError(err)
	is.Len(values, 1)
	is.True(math.IsNaN(values[0]))
}

func TestOperatorMathCeilWithPrecision(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		CeilWithPrecision(2)(Just(1.234, -1.234, 9.999)),
	)
	is.NoError(err)
	is.InDeltaSlice([]float64{1.24, -1.23, 10}, values, 1e-9)

	values, err = Collect(
		CeilWithPrecision(0)(Just(-2.2, 3.1)),
	)
	is.NoError(err)
	is.Equal([]float64{-2, 4}, values)

	values, err = Collect(
		CeilWithPrecision(-1)(Just(123.45, -123.45)),
	)
	is.NoError(err)
	is.InDeltaSlice([]float64{130, -120}, values, 1e-9)

	values, err = Collect(
		CeilWithPrecision(309)(Just(1.234, 1e-310)),
	)
	is.NoError(err)
	is.Len(values, 2)
	is.InDelta(1.234, values[0], 1e-15)
	is.NotEqual(math.Ceil(1.234), values[0])
	is.InDelta(1e-309, values[1], 1e-320)
	is.NotEqual(math.Ceil(1e-310), values[1])

	values, err = Collect(
		CeilWithPrecision(308)(Just(10.1, -10.1)),
	)
	is.NoError(err)
	is.Len(values, 2)
	is.InDelta(10.1, values[0], 1e-12)
	is.InDelta(-10.1, values[1], 1e-12)

	values, err = Collect(
		CeilWithPrecision(2)(Just(math.MaxFloat64 / 2)),
	)
	is.NoError(err)
	is.Len(values, 1)
	is.False(math.IsInf(values[0], 0))
	is.Equal(math.Ceil(math.MaxFloat64/2), values[0])

	values, err = Collect(
		CeilWithPrecision(-400)(Just(123.45, -123.45)),
	)
	is.NoError(err)
	is.Len(values, 2)
	is.True(math.IsInf(values[0], 1))
	is.Equal(0.0, values[1])

	values, err = Collect(
		CeilWithPrecision(-309)(Just(42.5, -42.5)),
	)
	is.NoError(err)
	is.Len(values, 2)
	is.True(math.IsInf(values[0], 1))
	is.Equal(0.0, values[1])

	values, err = Collect(
		CeilWithPrecision(3)(Just(math.Inf(1), math.Inf(-1), math.NaN())),
	)
	is.NoError(err)
	is.Len(values, 3)
	is.True(math.IsInf(values[0], 1))
	is.True(math.IsInf(values[1], -1))
	is.True(math.IsNaN(values[2]))

	values, err = Collect(
		CeilWithPrecision(math.MaxInt)(Just(1.2345, -6.789)),
	)
	is.NoError(err)
	is.InDeltaSlice([]float64{1.2345, -6.789}, values, 1e-12)

	values, err = Collect(
		CeilWithPrecision(math.MinInt + 1)(Just(42.5, -42.5, 0)),
	)
	is.NoError(err)
	is.Len(values, 3)
	is.True(math.IsInf(values[0], 1))
	is.Equal(0.0, values[1])
	is.Equal(0.0, values[2])
}

func TestOperatorMathCeilWithPrecisionMinInt(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		CeilWithPrecision(math.MinInt)(Just(42.5, -42.5, 0, math.Inf(1), math.Inf(-1), math.NaN())),
	)
	is.NoError(err)
	is.Len(values, 6)
	is.True(math.IsInf(values[0], 1))
	is.Equal(0.0, values[1])
	is.Equal(0.0, values[2])
	is.True(math.IsInf(values[3], 1))
	is.True(math.IsInf(values[4], -1))
	is.True(math.IsNaN(values[5]))
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
