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
	"io/fs"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOperatorTransformationMap(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	mapper := func(v int) int { return v * 2 }

	values, err := Collect(
		Map(mapper)(Just(1, 2, 3)),
	)
	is.Equal([]int{2, 4, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Map(mapper)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Map(mapper)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	// values, ctx, err := CollectWithContext(
	// 	context.WithValue(context.Background(), "foobar", 42),
	// 	Pipe1(
	// 		Just(1, 2, 3),
	// 		MapWithContext(func(ctx context.Context, n int) (context.Context, int) {
	// 			v := ctx.Value("foobar").(int)
	// 			is.Equal(42, v)

	// 			newCtx := context.WithValue(ctx, "foobar", v*2)
	// 			return newCtx, n * 2
	// 		}),
	// 	),
	// )
	// is.Equal([]int{2, 4, 6}, values)
	// is.Equal(42, ctx.Value("foobar").(int))
	// is.NoError(err)
}

func TestOperatorTransformationMapI(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	mapper := func(v int, _ int64) int { return v * 2 }

	values, err := Collect(
		MapI(mapper)(Just(1, 2, 3)),
	)
	is.Equal([]int{2, 4, 6}, values)
	is.NoError(err)

	values, err = Collect(
		MapI(func(v int, i int64) int {
			is.Equal(int(i), v)
			return v * 2
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{0, 2, 4, 6}, values)
	is.NoError(err)

	values, err = Collect(
		MapI(mapper)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		MapI(mapper)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationMapTo(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		MapTo[int](42)(Just(1, 2, 3)),
	)
	is.Equal([]int{42, 42, 42}, values)
	is.NoError(err)

	values, err = Collect(
		MapTo[int](42)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		MapTo[int](42)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationMapErr(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			MapErr(func(i int) (output int, err error) {
				return i * 2, nil
			}),
		),
	)
	is.Equal([]int{2, 4, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Of(1, 2, 3),
			MapErr(func(i int) (output int, err error) {
				if i == 3 {
					return 0, assert.AnError
				}

				return i * 2, nil
			}),
		),
	)
	is.Equal([]int{2, 4}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			Catch(func(err error) Observable[int] {
				is.EqualError(err, "ro.Observer: "+assert.AnError.Error())
				return Of(4, 5, 6)
			}),
		),
	)
	is.Equal([]int{1, 2, 4, 5, 6}, values)
	is.NoError(err)
}

func TestOperatorTransformationMapErrI(t *testing.T) { //nolint:paralleltest
	// @TODO: Implement tests
}

func TestOperatorTransformationFlatMap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			FlatMap(func(i int) Observable[int] {
				return Repeat(i, 3)
			}),
		),
	)
	is.Equal([]int{1, 1, 1, 2, 2, 2, 3, 3, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Of(1, 2, 3),
			FlatMap(func(i int) Observable[int] {
				if i == 3 {
					return Throw[int](assert.AnError)
				}

				return Repeat(i, 3)
			}),
		),
	)
	is.Equal([]int{1, 1, 1, 2, 2, 2}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			FlatMap(func(i int) Observable[int] {
				return Repeat(i, 3)
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationFlatten(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Just([]int{1, 2, 3}, []int{4, 5, 6}),
			Flatten[int](),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[[]int](),
			Flatten[int](),
		),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[[]int](assert.AnError),
			Flatten[int](),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationCast(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values1, err := Collect(
		Cast[any, int]()(Just[any](1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values1)
	is.NoError(err)

	values2, err := Collect(
		Cast[*fs.PathError, error]()(Just(&os.PathError{})),
	)
	is.Equal([]error{&os.PathError{}}, values2)
	is.NoError(err)

	values3, err := Collect(
		Cast[int, string]()(Just(1, 2, 3)),
	)
	is.Equal([]string{}, values3)
	is.EqualError(err, "ro.Cast: unable to cast int to string")

	values1, err = Collect(
		Cast[any, int]()(Empty[any]()),
	)
	is.Equal([]int{}, values1)
	is.NoError(err)

	values1, err = Collect(
		Cast[any, int]()(Throw[any](assert.AnError)),
	)
	is.Equal([]int{}, values1)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationScan(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	reduce := func(acc, item int) int { return acc + (item * 2) }

	values, err := Collect(
		Scan(reduce, 10)(Just(1, 2, 3)),
	)
	is.Equal([]int{12, 16, 22}, values)
	is.NoError(err)

	values, err = Collect(
		Scan(reduce, 10)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Scan(reduce, 10)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationScanI(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	reduce := func(acc, item int, _ int64) int { return acc + (item * 2) }

	values, err := Collect(
		ScanI(reduce, 10)(Just(1, 2, 3)),
	)
	is.Equal([]int{12, 16, 22}, values)
	is.NoError(err)

	values, err = Collect(
		ScanI(func(acc, item int, i int64) int {
			is.Equal(int(i), item)
			return acc + (item * 2)
		}, 10)(Just(0, 1, 2, 3)),
	)
	is.Equal([]int{10, 12, 16, 22}, values)
	is.NoError(err)

	values, err = Collect(
		ScanI(reduce, 10)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		ScanI(reduce, 10)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationGroupBy(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	odd := func(v int64) bool { return v%2 == 0 }

	values, err := Collect(
		Pipe2(
			RangeWithInterval(1, 8, 20*time.Millisecond),
			GroupBy(odd),
			MergeAll[int64](),
		),
	)
	is.Equal([]int64{1, 2, 3, 4, 5, 6, 7}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Empty[int64](),
			GroupBy(odd),
			MergeAll[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Throw[int64](assert.AnError),
			GroupBy(odd),
			MergeAll[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationBufferWhen(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(0, 5, 50*time.Millisecond),
			BufferWhen[int64](Interval(175*time.Millisecond)),
		),
	)
	is.Equal([][]int64{{0, 1, 2}, {3, 4}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			BufferWhen[int64](Interval(175*time.Millisecond)),
		),
	)
	is.Equal([][]int64{{}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 5, 50*time.Millisecond),
			BufferWhen[int64](Empty[int]()),
		),
	)
	is.Equal([][]int64{{}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			BufferWhen[int64](Interval(175*time.Millisecond)),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 5, 50*time.Millisecond),
			BufferWhen[int64](Throw[int64](assert.AnError)),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationBufferWithTimeOrCount(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		BufferWithTimeOrCount[int64](10, 100*time.Millisecond)(
			RangeWithInterval(1, 4, 20*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			NewObservable(func(destination Observer[int64]) Teardown {
				go func() {
					destination.Next(1)
					time.Sleep(150 * time.Millisecond)
					destination.Next(2)
					destination.Next(3)
					destination.Next(4)
					destination.Complete()
				}()

				return nil
			}),
			BufferWithTimeOrCount[int64](2, 100*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1}, {2, 3}, {4}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			NewObservable(func(destination Observer[int64]) Teardown {
				go func() {
					destination.Next(1)
					destination.Next(2)
					destination.Next(3)
					destination.Complete()
				}()

				return nil
			}),
			BufferWithTimeOrCount[int64](2, 50*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2}, {3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			NewObservable(func(destination Observer[int64]) Teardown {
				go func() {
					destination.Next(1)
					destination.Next(2)
					destination.Next(3)
					time.Sleep(175 * time.Millisecond)
					destination.Next(4)
					destination.Complete()
				}()

				return nil
			}),
			BufferWithTimeOrCount[int64](2, 50*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2}, {3}, {}, {}, {4}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			BufferWithTimeOrCount[int64](2, 50*time.Millisecond),
		),
	)
	is.Equal([][]int64{{}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			BufferWithTimeOrCount[int64](2, 50*time.Millisecond),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			NewObservable(func(destination Observer[int64]) Teardown {
				go func() {
					destination.Next(1)
					destination.Next(2)
					destination.Next(3)
					destination.Error(assert.AnError)
				}()

				return nil
			}),
			BufferWithTimeOrCount[int64](2, 50*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2}}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationBufferWithCount(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		BufferWithCount[int](1)(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1}, {2}, {3}}, values)
	is.NoError(err)

	values, err = Collect(
		BufferWithCount[int](2)(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2}, {3}}, values)
	is.NoError(err)

	values, err = Collect(
		BufferWithCount[int](3)(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		BufferWithCount[int](4)(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		BufferWithCount[int](4)(Empty[int]()),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	is.PanicsWithError("ro.BufferWithCount: size must be greater than 0", func() {
		BufferWithCount[int](0)(Just(1, 2, 3))
	})

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			BufferWithCount[int](2),
		),
	)
	is.Equal([][]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationBufferWithTime(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(1, 4, 50*time.Millisecond),
			BufferWithTime[int64](125*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2}, {3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(1, 4, 50*time.Millisecond),
			BufferWithTime[int64](300*time.Millisecond),
		),
	)
	is.Equal([][]int64{{1, 2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(1, 3, 200*time.Millisecond),
			BufferWithTime[int64](150*time.Millisecond),
		),
	)
	is.Equal([][]int64{{}, {1}, {2}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			BufferWithTime[int64](50*time.Millisecond),
		),
	)
	is.Equal([][]int64{{}}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			BufferWithTime[int64](50*time.Millisecond),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationWindowWhen(t *testing.T) { //nolint:paralleltest
	// @TODO: Implement tests
}

func TestOperatorTransformationSampleWhen(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1500*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe2(
			Timer(50*time.Millisecond),
			Map(func(v time.Duration) int64 { return 42 }),
			SampleWhen[int64](Interval(100*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Timer(100*time.Millisecond),
			Map(func(v time.Duration) int64 { return 42 }),
			SampleWhen[int64](Interval(50*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			RangeWithInterval(1, 8, 100*time.Millisecond),
			Delay[int64](50*time.Millisecond),
			SampleWhen[int64](Interval(300*time.Millisecond)),
		),
	)
	is.Equal([]int64{2, 5}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			SampleWhen[int64](Interval(20*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(1, 8, 20*time.Millisecond),
			SampleWhen[int64](Empty[int64]()),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			SampleWhen[int64](Interval(20*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(

		Pipe1(
			RangeWithInterval(1, 8, 20*time.Millisecond),
			SampleWhen[int64](Throw[int64](assert.AnError)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationSampleTime(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe2(
			Timer(50*time.Millisecond),
			Map(func(v time.Duration) int64 { return 42 }),
			SampleTime[int64](100*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Timer(100*time.Millisecond),
			Map(func(v time.Duration) int64 { return 42 }),
			SampleTime[int64](50*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			RangeWithInterval(1, 8, 100*time.Millisecond),
			Delay[int64](50*time.Millisecond),
			SampleWhen[int64](Interval(300*time.Millisecond)),
		),
	)
	is.Equal([]int64{2, 5}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			SampleTime[int64](20*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			SampleTime[int64](20*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationThrottleWhen(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(1, 8, 100*time.Millisecond),
			ThrottleWhen[int64](Interval(275*time.Millisecond)),
		),
	)
	is.Equal([]int64{3, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			ThrottleWhen[int64](Interval(25*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(1, 8, 50*time.Millisecond),
			ThrottleWhen[int64](Empty[int]()),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			ThrottleWhen[int64](Interval(25*time.Millisecond)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(1, 8, 50*time.Millisecond),
			ThrottleWhen[int64](Throw[int64](assert.AnError)),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorTransformationThrottleTime(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(1, 8, 50*time.Millisecond),
			ThrottleTime[int64](125*time.Millisecond),
		),
	)
	is.Equal([]int64{1, 4, 7}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			ThrottleTime[int64](25*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			ThrottleTime[int64](25*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}
