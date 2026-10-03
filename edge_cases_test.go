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

func TestEdgeCaseSubjectsInvalidBufferSize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.PanicsWithValue(ErrReplaySubjectWrongBufferSize, func() { NewReplaySubject[int](-2) })
	is.PanicsWithValue(ErrReplaySubjectWrongBufferSize, func() { NewReplaySubject[int](math.MinInt) })
	is.PanicsWithValue(ErrUnicastSubjectWrongBufferSize, func() { NewUnicastSubject[int](-2) })
	is.PanicsWithValue(ErrUnicastSubjectWrongBufferSize, func() { NewUnicastSubject[int](math.MinInt) })

	is.NotPanics(func() { NewReplaySubject[int](-1) })
	is.NotPanics(func() { NewReplaySubject[int](0) })
	is.NotPanics(func() { NewUnicastSubject[int](-1) })
	is.NotPanics(func() { NewUnicastSubject[int](0) })
}

func TestEdgeCaseSkipLastZero(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(SkipLast[int](0)(Just(1, 2, 3)))
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(SkipLast[int](0)(Empty[int]()))
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(SkipLast[int](0)(Throw[int](assert.AnError)))
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	is.PanicsWithValue(ErrSkipLastWrongCount, func() { SkipLast[int](-1) })
}

func TestEdgeCaseHugeCounts(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(TakeLast[int](math.MaxInt)(Just(1, 2, 3)))
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(SkipLast[int](math.MaxInt)(Just(1, 2, 3)))
	is.Equal([]int{}, values)
	is.NoError(err)

	chunks, err := Collect(BufferWithCount[int](math.MaxInt)(Just(1, 2, 3)))
	is.Equal([][]int{{1, 2, 3}}, chunks)
	is.NoError(err)
}

func TestEdgeCaseMinMaxNaN(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	nan := math.NaN()
	inputs := [][]float64{
		{nan, 1, 2},
		{1, nan, 2},
		{1, 2, nan},
		{nan},
	}

	for _, in := range inputs {
		values, err := Collect(Min[float64]()(FromSlice(in)))
		is.NoError(err)
		is.Len(values, 1)
		is.True(math.IsNaN(values[0]), "Min %v", in)

		values, err = Collect(Max[float64]()(FromSlice(in)))
		is.NoError(err)
		is.Len(values, 1)
		is.True(math.IsNaN(values[0]), "Max %v", in)
	}

	values, err := Collect(Max[float64]()(FromSlice([]float64{1, 3, 2})))
	is.Equal([]float64{3}, values)
	is.NoError(err)
}

func TestEdgeCaseMaxEmpty(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(Max[int]()(Empty[int]()))
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestEdgeCaseRangeExtremes(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Enumerating Range(0, MinInt64) is not feasible (Take cannot stop a synchronous source),
	// so check the loop condition directly at the boundaries.
	is.True(rangeHasNext(0, math.MinInt64, -1))
	is.True(rangeHasNext(math.MinInt64+1, math.MinInt64, -1))
	is.False(rangeHasNext(math.MinInt64, math.MinInt64, -1))
	is.True(rangeHasNext(math.MaxInt64-1, math.MaxInt64, 1))
	is.False(rangeHasNext(math.MaxInt64, math.MaxInt64, 1))

	values, err := Collect(Range(math.MaxInt64-2, math.MaxInt64))
	is.Equal([]int64{math.MaxInt64 - 2, math.MaxInt64 - 1}, values)
	is.NoError(err)

	values, err = Collect(Range(math.MinInt64+2, math.MinInt64))
	is.Equal([]int64{math.MinInt64 + 2, math.MinInt64 + 1}, values)
	is.NoError(err)
}

func TestEdgeCaseHeadTailEmptyMessages(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.EqualError(ErrHeadEmpty, "ro.Head: empty")
	is.EqualError(ErrTailEmpty, "ro.Tail: empty")
}
