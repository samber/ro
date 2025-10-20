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
	"strconv"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorCombiningMergeWith(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeWith1(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeWith2(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeWith3(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeWith4(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeWith5(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningMergeAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// sequential
	values, err := Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(6, 9, 100*time.Millisecond), // third
				RangeWithInterval(3, 6, 10*time.Millisecond),  // second
				Just[int64](0, 1, 2),                          // first
			),
		),
	)
	is.Equal([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8}, values)
	is.NoError(err)

	// parallel
	values, err = Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(0, 3, 100*time.Millisecond),
				RangeWithInterval(0, 3, 100*time.Millisecond),
				RangeWithInterval(0, 3, 100*time.Millisecond),
			),
		),
	)
	is.Equal([]int64{0, 0, 0, 1, 1, 1, 2, 2, 2}, values)
	is.NoError(err)

	// concurrent
	values, err = Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(0, 3, 200*time.Millisecond),
				Delay[int64](66*time.Millisecond)(RangeWithInterval(3, 6, 200*time.Millisecond)),
				Delay[int64](132*time.Millisecond)(RangeWithInterval(6, 9, 200*time.Millisecond)),
			),
		),
	)
	is.Equal([]int64{0, 3, 6, 1, 4, 7, 2, 5, 8}, values)
	is.NoError(err)

	values, err = Collect(
		MergeAll[int64]()(Empty[Observable[int64]]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		MergeAll[int64]()(Throw[Observable[int64]](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeMap(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(3, 7, 150*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{"3", "3", "3", "4", "4", "4", "4", "5", "5", "5", "5", "5", "6", "6", "6", "6", "6", "6"}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 2, 50*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), 3, 100*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{"0", "1", "0", "1", "0", "1"}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(3, 7, 30*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return Empty[string]()
			}),
		),
	)
	is.Equal([]string{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(3, 7, 30*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return Throw[string](assert.AnError)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningCombineLatestWith1(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// check type is in the right order
	values2, err := Collect(
		CombineLatestWith1[int](
			Of("42"),
		)(
			Of(42),
		),
	)
	is.Equal([]lo.Tuple2[int, string]{lo.T2(42, "42")}, values2)
	is.NoError(err)

	values1, err := Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 2, 50*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 75*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(0)), lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 2, 20*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 3, 10*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(2)), lo.T2(int64(1), int64(2))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Throw[int64](assert.AnError),
		)(
			Delay[int64](10 * time.Millisecond)(Of[int64](42)),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())

	values1, err = Collect(
		CombineLatestWith1[int64](
			Delay[int64](10 * time.Millisecond)(Throw[int64](assert.AnError)),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith2(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningCombineLatestWith3(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningCombineLatestWith4(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningCombineLatestAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](21),
				Of[int64](42),
			),
		),
	)
	is.Equal([][]int64{{21, 42}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 150*time.Millisecond),
				RangeWithInterval(0, 2, 100*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 0}, {0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 2, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 3, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 2}, {1, 2}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](42),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Empty[int64](),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Delay[int64](10*time.Millisecond)(Of[int64](42)),
				Throw[int64](assert.AnError),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](42),
				Delay[int64](10*time.Millisecond)(Throw[int64](assert.AnError)),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestAllAny(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](21),
				Of[any]("42"),
			),
		),
	)
	is.Equal([][]any{{21, "42"}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 150*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(0)}, {int64(0), int64(1)}, {int64(1), int64(1)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 20*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(1)}, {int64(1), int64(1)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 3, 10*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(2)}, {int64(1), int64(2)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](42),
				Empty[any](),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Empty[any](),
				Empty[any](),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Delay[any](10*time.Millisecond)(Of[any](42)),
				Throw[any](assert.AnError),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](42),
				Delay[any](10*time.Millisecond)(Throw[any](assert.AnError)),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningConcatWith(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningConcatAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ConcatAll[int]()(
			Just(
				Just(1, 2, 3),
				Just(4, 5, 6),
			),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		ConcatAll[int]()(Empty[Observable[int]]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		ConcatAll[int]()(Throw[Observable[int]](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningStartWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		StartWith(1, 2, 3)(Just(4, 5, 6)),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith(1, 2, 3)(Empty[int]()),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith(1, 2, 3)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningEndWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		EndWith(1, 2, 3)(Just(4, 5, 6)),
	)
	is.Equal([]int{4, 5, 6, 1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith(1, 2, 3)(Empty[int]()),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith(1, 2, 3)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningPairwise(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pairwise[int]()(Of(0)),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2}, {2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Empty[int]()),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([][]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningRaceWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1500*time.Millisecond)
	is := assert.New(t)

	// empty
	values, err := Collect(
		RaceWith[int]()(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// async
	values, err = Collect(
		RaceWith(
			Delay[int](150*time.Millisecond)(Just(4, 5, 6)),
			Delay[int](50*time.Millisecond)(Just(7, 8, 9)),
			Delay[int](200*time.Millisecond)(Just(10, 11, 12)),
		)(
			Delay[int](100 * time.Millisecond)(Just(1, 2, 3)),
		),
	)
	is.Equal([]int{7, 8, 9}, values)
	is.NoError(err)

	// sequential
	values, err = Collect(
		RaceWith(
			Just(4, 5, 6),
			Just(7, 8, 9),
			Just(10, 11, 12),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// mixed async + sequential
	values, err = Collect(
		RaceWith(
			Delay[int](150*time.Millisecond)(Just(4, 5, 6)),
			Just(4, 5, 6),
			Delay[int](200*time.Millisecond)(Just(10, 11, 12)),
		)(
			Delay[int](100 * time.Millisecond)(Just(1, 2, 3)),
		),
	)
	is.Equal([]int{4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Race(Empty[int](), Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Race[int](),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Race(
			Delay[int](250*time.Millisecond)(Just(1, 2, 3)),
			Delay[int](25*time.Millisecond)(Throw[int](assert.AnError)),
			Delay[int](250*time.Millisecond)(Just(7, 8, 9)),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 10),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Of[int64](42),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith1(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningZipWith2(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningZipWith3(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningZipWith4(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningZipWith5(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}

func TestOperatorCombiningZipAll(t *testing.T) { //nolint:paralleltest
	// @TODO: implement
}
