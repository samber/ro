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
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	// parallel merge of synchronous sources: all 3 sources emit 3 values
	values, err := Collect(
		MergeWith[int64](
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
		)(
			Just[int64](0, 1, 2),
		),
	)
	is.Len(values, 9)
	is.NoError(err)
	is.ElementsMatch(values, []int64{0, 1, 2, 3, 4, 5, 6, 7, 8})

	// parallel: same values from 3 sources
	values, err = Collect(
		MergeWith[int64](
			Just[int64](0),
			Just[int64](0),
		)(
			Just[int64](0),
		),
	)
	is.Equal([]int64{0, 0, 0}, values)
	is.NoError(err)

	// empty source
	values, err = Collect(
		MergeWith[int64]()(
			Empty[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	// error source
	values, err = Collect(
		MergeWith[int64]()(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith1(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// MergeWith is a parallel merge: all sources emit concurrently,
	// so use ElementsMatch to verify all values are present regardless of order.
	// Just(0,1,2) and RangeWithInterval(6,9,100ms): values 0,1,2 immediate, 6,7,8 delayed
	values, err := Collect(
		MergeWith1[int64](
			RangeWithInterval(6, 9, 100*time.Millisecond),
		)(
			Just[int64](0, 1, 2),
		),
	)
	is.Len(values, 6)
	is.ElementsMatch(values, []int64{0, 1, 2, 6, 7, 8})
	is.NoError(err)

	// parallel: two identical sources each emit 0,1,2 → 6 values total
	values, err = Collect(
		MergeWith1[int64](
			RangeWithInterval(0, 3, 100*time.Millisecond),
		)(
			RangeWithInterval(0, 3, 100*time.Millisecond),
		),
	)
	is.Len(values, 6)
	is.ElementsMatch(values, []int64{0, 0, 1, 1, 2, 2})
	is.NoError(err)

	// concurrent: Just(0) emits immediately, delayed source emits 0,1,2 after 100ms
	values, err = Collect(
		MergeWith1[int64](
			Delay[int64](100 * time.Millisecond)(RangeWithInterval(0, 3, 200*time.Millisecond)),
		)(
			Just[int64](0),
		),
	)
	is.Len(values, 4)
	is.ElementsMatch(values, []int64{0, 0, 1, 2})
	is.NoError(err)

	values, err = Collect(
		MergeWith1[int64](
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	// MergeWith subscribes concurrently; if Throw errors first,
	// Just(42) may not have emitted yet — only check error
	_, err = Collect(
		MergeWith1[int64](
			Throw[int64](assert.AnError),
		)(
			Just[int64](42),
		),
	)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith2(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith2[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
		)(
			Just[int64](6, 7, 8),
		),
	)
	is.Len(values, 9)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith2[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
		)(
			Just[int64](42),
		),
	)
	is.Empty(values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith3[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
		)(
			Just[int64](9, 10, 11),
		),
	)
	is.Len(values, 12)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith3[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Empty(values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith4[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
			Just[int64](9, 10, 11),
		)(
			Just[int64](12, 13, 14),
		),
	)
	is.Len(values, 15)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith4[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Empty(values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith5(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith5[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
			Just[int64](9, 10, 11),
			Just[int64](12, 13, 14),
		)(
			Just[int64](15, 16, 17),
		),
	)
	is.Len(values, 18)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith5[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith5[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Empty(values)
	is.EqualError(err, assert.AnError.Error())
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
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// check type is in the right order
	values2, err := Collect(
		CombineLatestWith[int](
			Of("42"),
		)(
			Of(42),
		),
	)
	is.Equal([]lo.Tuple2[int, string]{lo.T2(42, "42")}, values2)
	is.NoError(err)

	values1, err := Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 2, 50*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 75*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(0)), lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 2, 20*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 3, 10*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(2)), lo.T2(int64(1), int64(2))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Throw[int64](assert.AnError),
		)(
			Delay[int64](10 * time.Millisecond)(Of[int64](42)),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())

	values1, err = Collect(
		CombineLatestWith[int64](
			Delay[int64](10 * time.Millisecond)(Throw[int64](assert.AnError)),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())
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
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// CombineLatest emits a tuple whenever any source emits after all sources have emitted at least once
	// With Of (single-value) sources, only one tuple is produced
	values, err := Collect(
		CombineLatestWith2[int64](
			Of[int64](1),
			Of[int64](2),
		)(
			Of[int64](3),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC) — source is the function argument, not the variadic params
	is.Equal([]lo.Tuple3[int64, int64, int64]{lo.T3(int64(3), int64(1), int64(2))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith2[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestWith3[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
		)(
			Of[int64](4),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC, argD) — source is the function argument
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{lo.T4(int64(4), int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith3[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Of[int64](5),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC, argD, argE) — source is the function argument
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{lo.T5(int64(5), int64(1), int64(2), int64(3), int64(4))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith4[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
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
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// sequential concatenation
	values, err := Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	// concatenate multiple observables
	values, err = Collect(
		ConcatWith(
			Just(4, 5),
			Just(6, 7),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6, 7}, values)
	is.NoError(err)

	// empty source
	values, err = Collect(
		ConcatWith(
			Just(1, 2, 3),
		)(
			Empty[int](),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// empty additional observable
	values, err = Collect(
		ConcatWith(
			Empty[int](),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// error propagation from source: source Throw is the first inner observable,
	// so ConcatAll errors immediately without emitting values.
	values, err = Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	// error propagation from additional observable
	values, err = Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	// error from additional observable after source completes
	values, err = Collect(
		ConcatWith(
			Throw[int](assert.AnError),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())
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
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith1[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 10),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Of[int64](42),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith2(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith2[int64](
			Skip[int64](1)(Range(0, 4)),
			Skip[int64](2)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{lo.T3(int64(0), int64(1), int64(2)), lo.T3(int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Range(0, 4),
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Of[int64](42),
			Of[int64](99),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith3[int64](
			Skip[int64](1)(Range(0, 4)),
			Skip[int64](2)(Range(0, 4)),
			Skip[int64](3)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{lo.T4(int64(0), int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith3[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Of[int64](5),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{lo.T5(int64(5), int64(1), int64(2), int64(3), int64(4))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith5(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// ZipWith5 is strict — all sources must have values; tuple is (source, arg1, arg2, arg3, arg4, arg5)
	values, err := Collect(
		ZipWith5[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
			Of[int64](5),
		)(
			Of[int64](0),
		),
	)
	is.Len(values, 1)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{lo.T6(int64(0), int64(1), int64(2), int64(3), int64(4), int64(5))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith5[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith5[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
			Of[int64](5),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](21),
				Of[int64](42),
			),
		),
	)
	is.Equal([][]int64{{21, 42}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 150*time.Millisecond),
				RangeWithInterval(0, 2, 100*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 0}, {0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 2, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 3, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 2}, {1, 2}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](42),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Empty[int64](),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Delay[int64](10*time.Millisecond)(Of[int64](42)),
				Throw[int64](assert.AnError),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](42),
				Delay[int64](10*time.Millisecond)(Throw[int64](assert.AnError)),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}
