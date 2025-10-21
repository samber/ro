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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOperatorFilterFilter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(x int) bool {
		return x%2 == 0
	}

	values, err := Collect(
		Filter(predicate)(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{0, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Filter(predicate)(Just(1, 2)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		Filter(predicate)(Just(1, -1)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Filter(predicate)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Filter(predicate)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterFilterI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(x int, i int64) bool {
		return x%2 == 0
	}

	values, err := Collect(
		FilterI(predicate)(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{0, 2}, values)
	is.NoError(err)

	values, err = Collect(
		FilterI(func(x int, i int64) bool {
			is.Equal(int(i), x)
			return x%2 == 0
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{0, 2}, values)
	is.NoError(err)

	values, err = Collect(
		FilterI(predicate)(Just(1, 2)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		FilterI(predicate)(Just(1, -1)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		FilterI(predicate)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		FilterI(predicate)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterDistinct(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Distinct[int]()(Just(0, 1, 2)),
	)
	is.Equal([]int{0, 1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Distinct[int]()(Just(0, 1, 2, 0, 1, 2)),
	)
	is.Equal([]int{0, 1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Distinct[int]()(Just(0, 1, 2, 2, 1, 0)),
	)
	is.Equal([]int{0, 1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Distinct[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Distinct[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterDistinctBy(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	type user struct {
		id   int
		name string
	}

	obs := Pipe1(
		Just(
			user{id: 1, name: "John"},
			user{id: 2, name: "Jane"},
			user{id: 1, name: "John"},
			user{id: 3, name: "Jim"},
		),
		DistinctBy(func(item user) int {
			return item.id
		}),
	)
	values, err := Collect(obs)
	is.Equal([]user{{id: 1, name: "John"}, {id: 2, name: "Jane"}}, values)
	is.NoError(err)

	// empty
	values, err = Collect(
		DistinctBy(func(item user) int {
			return item.id
		})(Empty[user]()),
	)
	is.Equal([]user{}, values)
	is.NoError(err)

	// error
	values, err = Collect(
		DistinctBy(func(item user) int {
			return item.id
		})(Throw[user](assert.AnError)),
	)
	is.Equal([]user{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterIgnoreElements(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		IgnoreElements[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		IgnoreElements[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		IgnoreElements[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterSkip(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Skip[int](2)(Just(1, 2, 3)),
	)
	is.Equal([]int{3}, values)
	is.NoError(err)

	values, err = Collect(
		Skip[int](0)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Skip[int](42)(Just(1, 2, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Skip[int](0)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Skip[int](0)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterSkipWhile(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(v int64) bool {
		return v > 5
	}

	values, err := Collect(
		SkipWhile(predicate)(Range(1, 10)),
	)
	is.Equal([]int64{6, 7, 8, 9}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhile(predicate)(Range(1, 3)),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhile(predicate)(Just[int64](10, 11, 12)),
	)
	is.Equal([]int64{10, 11, 12}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhile(predicate)(Empty[int64]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhile(predicate)(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterSkipWhileI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(v, i int64) bool {
		return v > 5
	}

	values, err := Collect(
		SkipWhileI(predicate)(Range(1, 10)),
	)
	is.Equal([]int64{6, 7, 8, 9}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhileI(func(v, i int64) bool {
			is.Equal(v, i)
			return v > 5
		})(Range(0, 10)),
	)
	is.Equal([]int64{6, 7, 8, 9}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhileI(predicate)(Range(1, 3)),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhileI(predicate)(Just[int64](10, 11, 12)),
	)
	is.Equal([]int64{10, 11, 12}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhileI(predicate)(Empty[int64]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipWhileI(predicate)(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterSkipLast(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		SkipLast[int64](2)(Range(1, 5)),
	)
	is.Equal([]int64{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		SkipLast[int64](10)(Range(1, 5)),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipLast[int64](10)(Empty[int64]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		SkipLast[int64](10)(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterSkipUntil(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(0, 5, 50*time.Millisecond),
			SkipUntil[int64](Interval(125*time.Millisecond)),
		),
	)
	is.Equal([]int64{2, 3, 4}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 50*time.Millisecond),
			SkipUntil[int64](Interval(500*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			SkipUntil[int64](Interval(10*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 10*time.Millisecond),
			SkipUntil[int64](Empty[int64]()),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			SkipUntil[int64](Interval(10*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 10*time.Millisecond),
			SkipUntil[int64](Throw[int64](assert.AnError)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)
}

func TestOperatorFilterTake(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Take[int](10)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Take[int](2)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Take[int](0)(Just(1, 2, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Take[int](42)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Take[int](42)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterTakeWhile(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(v int) bool {
		return v < 3
	}

	values, err := Collect(
		TakeWhile(predicate)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhile(predicate)(Just(1, 2)),
	)
	is.Equal([]int{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhile(predicate)(Just(1)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhile(predicate)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhile(predicate)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterTakeWhileI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	predicate := func(v int, i int64) bool {
		return v < 3
	}

	values, err := Collect(
		TakeWhileI(predicate)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhileI(func(v int, i int64) bool {
			is.Equal(v, int(i))
			return v < 3
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{0, 1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhileI(predicate)(Just(1, 2)),
	)
	is.Equal([]int{1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhileI(predicate)(Just(1)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhileI(predicate)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		TakeWhileI(predicate)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterTakeLast(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		TakeLast[int](10)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		TakeLast[int](2)(Just(1, 2, 3)),
	)
	is.Equal([]int{2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		TakeLast[int](0)(Just(1, 2, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		TakeLast[int](42)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		TakeLast[int](42)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	is.Panics(func() {
		values, err = Collect(
			TakeLast[int](-1)(Just(1, 2, 3)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})
}

func TestOperatorFilterTakeUntil(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(0, 5, 20*time.Millisecond),
			TakeUntil[int64](Interval(50*time.Millisecond)),
		),
	)
	is.Equal([]int64{0, 1}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 20*time.Millisecond),
			TakeUntil[int64](Interval(10*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			TakeUntil[int64](Interval(10*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 10*time.Millisecond),
			TakeUntil[int64](Empty[int64]()),
		),
	)
	is.Equal([]int64{0, 1, 2}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			TakeUntil[int64](Interval(10*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 3, 10*time.Millisecond),
			TakeUntil[int64](Throw[int64](assert.AnError)),
		),
	)
	is.Equal([]int64{0, 1, 2}, values)
	is.NoError(err)
}

func TestOperatorFilterHead(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Head[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Head[int]()(Just(1)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Head[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrFirstEmpty.Error())

	values, err = Collect(
		Head[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterTail(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Tail[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{3}, values)
	is.NoError(err)

	values, err = Collect(
		Tail[int]()(Just(1)),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Tail[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrLastEmpty.Error())

	values, err = Collect(
		Tail[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterFirst(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Just(1, 2, 3),
			First(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1),
			First(func(item int) bool {
				return item > 0
			}),
		),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1),
			First(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrFirstEmpty.Error())

	values, err = Collect(
		Pipe1(
			Empty[int](),
			First(func(item int) bool {
				return item > 2
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrFirstEmpty.Error())

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			First(func(item int) bool {
				return item > 2
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterLast(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Just(1, 2, 3),
			Last(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1),
			Last(func(item int) bool {
				return item > 0
			}),
		),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1),
			Last(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrLastEmpty.Error())

	values, err = Collect(
		Pipe1(
			Empty[int](),
			Last(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrLastEmpty.Error())

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			Last(func(item int) bool {
				return item > 1
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterElementAt(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.PanicsWithError(
		ErrElementAtWrongNth.Error(),
		func() {
			_ = ElementAt[int](-42)(Just(1, 2, 3))
		},
	)

	values, err := Collect(
		ElementAt[int](1)(Just(1, 2, 3)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		ElementAt[int](2)(Just(1)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrElementAtNotFound.Error())

	values, err = Collect(
		ElementAt[int](42)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, ErrElementAtNotFound.Error())

	values, err = Collect(
		ElementAt[int](42)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorFilterElementAtOrDefault(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.PanicsWithError(
		ErrElementAtOrDefaultWrongNth.Error(),
		func() {
			_ = ElementAtOrDefault(-42, 42)(Just(1, 2, 3))
		},
	)

	values, err := Collect(
		ElementAtOrDefault(1, 100)(Just(1, 2, 3)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		ElementAtOrDefault(2, 100)(Just(1)),
	)
	is.Equal([]int{100}, values)
	is.NoError(err)

	values, err = Collect(
		ElementAtOrDefault(42, 100)(Empty[int]()),
	)
	is.Equal([]int{100}, values)
	is.NoError(err)

	values, err = Collect(
		ElementAtOrDefault(42, 100)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}
