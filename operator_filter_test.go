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
	"context"
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOperatorFilterFilter(t *testing.T) {
	t.Parallel()
	predicate := func(x int) bool {
		return x%2 == 0
	}

	t.Run("filters even values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Filter(predicate)(Just(0, 1, 2, 3)),
		)
		is.Equal([]int{0, 2}, values)
		is.NoError(err)
	})

	t.Run("keeps the matching tail", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Filter(predicate)(Just(1, 2)),
		)
		is.Equal([]int{2}, values)
		is.NoError(err)
	})

	t.Run("no value matches", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Filter(predicate)(Just(1, -1)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Filter(predicate)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Filter(predicate)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterFilterI(t *testing.T) {
	t.Parallel()
	predicate := func(x int, i int64) bool {
		return x%2 == 0
	}

	t.Run("filters even values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(predicate)(Just(0, 1, 2, 3)),
		)
		is.Equal([]int{0, 2}, values)
		is.NoError(err)
	})

	t.Run("index matches item position", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(func(x int, i int64) bool {
				is.Equal(int(i), x)
				return x%2 == 0
			})(Just(0, 1, 2, 3)),
		)
		is.Equal([]int{0, 2}, values)
		is.NoError(err)
	})

	t.Run("keeps the matching tail", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(predicate)(Just(1, 2)),
		)
		is.Equal([]int{2}, values)
		is.NoError(err)
	})

	t.Run("no value matches", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(predicate)(Just(1, -1)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(predicate)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			FilterI(predicate)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterDistinct(t *testing.T) {
	t.Parallel()

	t.Run("all values unique", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Distinct[int]()(Just(0, 1, 2)),
		)
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
	})

	t.Run("repeated sequence", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Distinct[int]()(Just(0, 1, 2, 0, 1, 2)),
		)
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
	})

	t.Run("mirrored sequence", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Distinct[int]()(Just(0, 1, 2, 2, 1, 0)),
		)
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Distinct[int]()(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Distinct[int]()(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterDistinctBy(t *testing.T) {
	t.Parallel()
	type user struct {
		id   int
		name string
	}

	t.Run("distinct by key", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

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
		is.Equal([]user{{id: 1, name: "John"}, {id: 2, name: "Jane"}, {id: 3, name: "Jim"}}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			DistinctBy(func(item user) int {
				return item.id
			})(Empty[user]()),
		)
		is.Equal([]user{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			DistinctBy(func(item user) int {
				return item.id
			})(Throw[user](assert.AnError)),
		)
		is.Equal([]user{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterIgnoreElements(t *testing.T) {
	t.Parallel()

	t.Run("values are ignored", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			IgnoreElements[int]()(Just(1, 2, 3)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			IgnoreElements[int]()(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			IgnoreElements[int]()(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterSkip(t *testing.T) {
	t.Parallel()

	t.Run("skips two values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Skip[int](2)(Just(1, 2, 3)),
		)
		is.Equal([]int{3}, values)
		is.NoError(err)
	})

	t.Run("skips zero values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Skip[int](0)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("skips more than available", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Skip[int](42)(Just(1, 2, 3)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Skip[int](0)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Skip[int](0)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterSkipWhile(t *testing.T) {
	t.Parallel()
	predicate := func(v int64) bool {
		return v <= 5
	}

	t.Run("skips leading matches", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhile(predicate)(Range(1, 10)),
		)
		is.Equal([]int64{6, 7, 8, 9}, values)
		is.NoError(err)
	})

	t.Run("skips every value", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhile(predicate)(Range(1, 3)),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("nothing to skip", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhile(predicate)(Just[int64](10, 11, 12)),
		)
		is.Equal([]int64{10, 11, 12}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhile(predicate)(Empty[int64]()),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhile(predicate)(Throw[int64](assert.AnError)),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterSkipWhileI(t *testing.T) {
	t.Parallel()
	predicate := func(v, i int64) bool {
		return v <= 5
	}

	t.Run("skips leading matches", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(predicate)(Range(1, 10)),
		)
		is.Equal([]int64{6, 7, 8, 9}, values)
		is.NoError(err)
	})

	t.Run("index matches item position", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(func(v, i int64) bool {
				is.Equal(v, i)
				return v <= 5
			})(Range(0, 10)),
		)
		is.Equal([]int64{6, 7, 8, 9}, values)
		is.NoError(err)
	})

	t.Run("skips every value", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(predicate)(Range(1, 3)),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("nothing to skip", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(predicate)(Just[int64](10, 11, 12)),
		)
		is.Equal([]int64{10, 11, 12}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(predicate)(Empty[int64]()),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipWhileI(predicate)(Throw[int64](assert.AnError)),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterSkipLast(t *testing.T) {
	t.Parallel()

	t.Run("skips last two values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipLast[int64](2)(Range(1, 5)),
		)
		is.Equal([]int64{1, 2}, values)
		is.NoError(err)
	})

	t.Run("count exceeds source length", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipLast[int64](10)(Range(1, 5)),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipLast[int64](10)(Empty[int64]()),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			SkipLast[int64](10)(Throw[int64](assert.AnError)),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("zero count", func(t *testing.T) {
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
	})

	t.Run("huge count on TakeLast and SkipLast", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(TakeLast[int](math.MaxInt)(Just(1, 2, 3)))
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(SkipLast[int](math.MaxInt)(Just(1, 2, 3)))
		is.Equal([]int{}, values)
		is.NoError(err)
	})
}

func TestOperatorFilterSkipUntil(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)

	t.Run("signal opens the gate mid-stream", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 5, 50*time.Millisecond),
				SkipUntil[int64](Interval(125*time.Millisecond)),
			),
		)
		is.Equal([]int64{2, 3, 4}, values)
		is.NoError(err)
	})

	t.Run("signal fires after the source ends", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 50*time.Millisecond),
				SkipUntil[int64](Interval(500*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Empty[int64](),
				SkipUntil[int64](Interval(10*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("empty signal", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 10*time.Millisecond),
				SkipUntil[int64](Empty[int64]()),
			),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("source error", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int64](assert.AnError),
				SkipUntil[int64](Interval(10*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("signal error", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 10*time.Millisecond),
				SkipUntil[int64](Throw[int64](assert.AnError)),
			),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterSkipUntil_signalOrder(t *testing.T) {
	t.Parallel()

	t.Run("synchronous signal opens the gate", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		// The signal is subscribed before the source: a synchronous signal opens
		// the gate before a synchronous source emits.
		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				SkipUntil[int](Just("go")),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("completing signal never opens the gate", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				SkipUntil[int](Empty[string]()),
			),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("signal error terminates the stream", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				SkipUntil[int](Throw[string](assert.AnError)),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("source error wins over silent signal", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int](assert.AnError),
				SkipUntil[int](Never()),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterTake(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	t.Run("take more than available", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Take[int](10)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("take two values", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Take[int](2)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2}, values)
		is.NoError(err)
	})

	t.Run("take zero values", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Take[int](0)(Just(1, 2, 3)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Take[int](42)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Take[int](42)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(Take[int](2)(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0, 1}, values)
		is.NoError(err)
		is.Equal(int64(2), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(Take[int](1)(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0}, values)
		is.NoError(err)
		is.Equal(int64(1), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(Take[int64](2)(Range(0, math.MaxInt64)))
		is.Equal([]int64{0, 1}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(Take[int](3)(tickingAsyncSource(&tornDown)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterTakeWhile(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	predicate := func(v int) bool {
		return v < 3
	}

	t.Run("stops at first failing value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhile(predicate)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2}, values)
		is.NoError(err)
	})

	t.Run("predicate holds for every value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhile(predicate)(Just(1, 2)),
		)
		is.Equal([]int{1, 2}, values)
		is.NoError(err)
	})

	t.Run("single value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhile(predicate)(Just(1)),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhile(predicate)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhile(predicate)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(TakeWhile(func(v int) bool { return v < 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(TakeWhile(func(v int) bool { return false })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{}, values)
		is.NoError(err)
		is.Equal(int64(1), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(TakeWhileWithContext(func(ctx context.Context, v int) (context.Context, bool) { return ctx, v < 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(TakeWhile(func(v int64) bool { return v < 2 })(Range(0, math.MaxInt64)))
		is.Equal([]int64{0, 1}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(TakeWhile(func(v int) bool { return v < 3 })(tickingAsyncSource(&tornDown)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterTakeWhileI(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	predicate := func(v int, i int64) bool {
		return v < 3
	}

	t.Run("stops at first failing value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(predicate)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2}, values)
		is.NoError(err)
	})

	t.Run("index matches item position", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(func(v int, i int64) bool {
				is.Equal(v, int(i))
				return v < 3
			})(Just(0, 1, 2, 3)),
		)
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
	})

	t.Run("predicate holds for every value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(predicate)(Just(1, 2)),
		)
		is.Equal([]int{1, 2}, values)
		is.NoError(err)
	})

	t.Run("single value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(predicate)(Just(1)),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(predicate)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeWhileI(predicate)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(TakeWhileI(func(v int, i int64) bool { return i < 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(TakeWhileIWithContext(func(ctx context.Context, v int, i int64) (context.Context, bool) { return ctx, i < 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(TakeWhileI(func(v int, i int64) bool { return i < 3 })(tickingAsyncSource(&tornDown)))
		is.Equal([]int{0, 1, 2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterTakeLast(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 10*time.Millisecond)

	t.Run("count exceeds source length", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeLast[int](10)(Just(1, 2, 3)),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("last two values", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeLast[int](2)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 3}, values)
		is.NoError(err)
	})

	t.Run("zero count", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeLast[int](0)(Just(1, 2, 3)),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeLast[int](42)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			TakeLast[int](42)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("negative count panics", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		is.Panics(func() {
			values, err := Collect(
				TakeLast[int](-1)(Just(1, 2, 3)),
			)
			is.Equal([]int{}, values)
			is.NoError(err)
		})
	})
}

func TestOperatorFilterTakeUntil(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)

	t.Run("signal fires mid-stream", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 5, 20*time.Millisecond),
				TakeUntil[int64](Interval(50*time.Millisecond)),
			),
		)
		is.Equal([]int64{0, 1}, values)
		is.NoError(err)
	})

	t.Run("signal fires before the first value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 20*time.Millisecond),
				TakeUntil[int64](Interval(10*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Empty[int64](),
				TakeUntil[int64](Interval(10*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.NoError(err)
	})

	t.Run("empty signal", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 10*time.Millisecond),
				TakeUntil[int64](Empty[int64]()),
			),
		)
		is.Equal([]int64{0, 1, 2}, values)
		is.NoError(err)
	})

	t.Run("source error", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int64](assert.AnError),
				TakeUntil[int64](Interval(10*time.Millisecond)),
			),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("signal error", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				RangeWithInterval(0, 3, 10*time.Millisecond),
				TakeUntil[int64](Throw[int64](assert.AnError)),
			),
		)
		is.Equal([]int64{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterTakeUntil_signalOrder(t *testing.T) {
	t.Parallel()

	t.Run("synchronous signal completes before the source emits", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		// The signal is subscribed before the source: a synchronous signal
		// completes the stream before a synchronous source emits.
		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				TakeUntil[int](Just("stop")),
			),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("completing signal never stops the stream", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				TakeUntil[int](Empty[string]()),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("signal error terminates the stream", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				TakeUntil[int](Throw[string](assert.AnError)),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("source error wins over silent signal", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int](assert.AnError),
				TakeUntil[int](Never()),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterHead(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	t.Run("first of several values", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Head[int]()(Just(1, 2, 3)),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("single value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Head[int]()(Just(1)),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Head[int]()(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrHeadEmpty.Error())
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Head[int]()(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(Head[int]()(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0}, values)
		is.NoError(err)
		is.Equal(int64(1), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(Head[int64]()(Repeat(int64(7), math.MaxInt64)))
		is.Equal([]int64{7}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(Head[int]()(tickingAsyncSource(&tornDown)))
		is.Equal([]int{0}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterTail(t *testing.T) {
	t.Parallel()

	t.Run("last of several values", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Tail[int]()(Just(1, 2, 3)),
		)
		is.Equal([]int{3}, values)
		is.NoError(err)
	})

	t.Run("single value", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Tail[int]()(Just(1)),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Tail[int]()(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrTailEmpty.Error())
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Tail[int]()(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterFirst(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	t.Run("first match", func(t *testing.T) { //nolint:paralleltest
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
	})

	t.Run("match on the only value", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1),
				First(func(item int) bool {
					return item > 0
				}),
			),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("no value matches", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1),
				First(func(item int) bool {
					return item > 1
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrFirstEmpty.Error())
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Empty[int](),
				First(func(item int) bool {
					return item > 2
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrFirstEmpty.Error())
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int](assert.AnError),
				First(func(item int) bool {
					return item > 2
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(First(func(v int) bool { return v == 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(FirstI(func(v int, i int64) bool { return i == 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(FirstWithContext(func(ctx context.Context, v int) (context.Context, bool) { return ctx, v == 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(FirstIWithContext(func(ctx context.Context, v int, i int64) (context.Context, bool) { return ctx, i == 3 })(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(First(func(v int64) bool { return v == 2 })(Range(0, math.MaxInt64)))
		is.Equal([]int64{2}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(First(func(v int) bool { return v == 2 })(tickingAsyncSource(&tornDown)))
		is.Equal([]int{2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterLast(t *testing.T) {
	t.Parallel()

	t.Run("last match", func(t *testing.T) {
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
	})

	t.Run("match on the only value", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1),
				Last(func(item int) bool {
					return item > 0
				}),
			),
		)
		is.Equal([]int{1}, values)
		is.NoError(err)
	})

	t.Run("no value matches", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Just(1),
				Last(func(item int) bool {
					return item > 1
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrLastEmpty.Error())
	})

	t.Run("empty source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Empty[int](),
				Last(func(item int) bool {
					return item > 1
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrLastEmpty.Error())
	})

	t.Run("error propagation", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := Collect(
			Pipe1(
				Throw[int](assert.AnError),
				Last(func(item int) bool {
					return item > 1
				}),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestOperatorFilterElementAt(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	t.Run("negative index panics", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		is.PanicsWithError(
			ErrElementAtWrongNth.Error(),
			func() {
				_ = ElementAt[int](-42)(Just(1, 2, 3))
			},
		)
	})

	t.Run("element at index", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAt[int](1)(Just(1, 2, 3)),
		)
		is.Equal([]int{2}, values)
		is.NoError(err)
	})

	t.Run("index beyond source length", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAt[int](2)(Just(1)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrElementAtNotFound.Error())
	})

	t.Run("empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAt[int](42)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, ErrElementAtNotFound.Error())
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAt[int](42)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(ElementAt[int](3)(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))

		emitted = 0
		values, err = Collect(ElementAt[int](0)(countingSyncSource(1000, &emitted)))
		is.Equal([]int{0}, values)
		is.NoError(err)
		is.Equal(int64(1), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(ElementAt[int64](2)(Range(0, math.MaxInt64)))
		is.Equal([]int64{2}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(ElementAt[int](2)(tickingAsyncSource(&tornDown)))
		is.Equal([]int{2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}

func TestOperatorFilterElementAtOrDefault(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)

	t.Run("negative index panics", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		is.PanicsWithError(
			ErrElementAtOrDefaultWrongNth.Error(),
			func() {
				_ = ElementAtOrDefault(-42, 42)(Just(1, 2, 3))
			},
		)
	})

	t.Run("element at index", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAtOrDefault(1, 100)(Just(1, 2, 3)),
		)
		is.Equal([]int{2}, values)
		is.NoError(err)
	})

	t.Run("default when index is beyond source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAtOrDefault(2, 100)(Just(1)),
		)
		is.Equal([]int{100}, values)
		is.NoError(err)
	})

	t.Run("default on empty source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAtOrDefault(42, 100)(Empty[int]()),
		)
		is.Equal([]int{100}, values)
		is.NoError(err)
	})

	t.Run("error propagation", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		values, err := Collect(
			ElementAtOrDefault(42, 100)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("early completion stops a synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var emitted int64

		values, err := Collect(ElementAtOrDefault(3, -1)(countingSyncSource(1000, &emitted)))
		is.Equal([]int{3}, values)
		is.NoError(err)
		is.Equal(int64(4), atomic.LoadInt64(&emitted))
	})

	t.Run("unbounded synchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		infinite, err := Collect(ElementAtOrDefault(2, int64(-1))(Range(0, math.MaxInt64)))
		is.Equal([]int64{2}, infinite)
		is.NoError(err)
	})

	t.Run("early completion tears down an asynchronous source", func(t *testing.T) { //nolint:paralleltest
		is := assert.New(t)

		var tornDown int32

		values, err := Collect(ElementAtOrDefault(2, -1)(tickingAsyncSource(&tornDown)))
		is.Equal([]int{2}, values)
		is.NoError(err)
		is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
	})
}
