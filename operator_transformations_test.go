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
	"io/fs"
	"math"
	"os"
	"runtime"
	"sync/atomic"
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

// Unsubscribing while the source is still emitting new and existing keys must
// neither race on the group registry nor leave a group uncompleted.
func TestOperatorTransformationGroupByTeardownRacesInFlightValues(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	const (
		rounds    = 200
		keys      = 8
		emissions = 2000
	)

	for r := 0; r < rounds; r++ {
		source := NewPublishSubject[int]()

		var open int64 // groups emitted but not yet completed; atomic.Int64 needs Go 1.19

		// Yielding in the iteratee keeps values in flight while teardown runs.
		iteratee := func(v int) int {
			runtime.Gosched()
			return v % keys
		}

		sub := GroupBy(iteratee)(source).Subscribe(
			OnNext(func(group Observable[int]) {
				atomic.AddInt64(&open, 1)
				group.Subscribe(NewObserver(
					func(int) {},
					func(error) { atomic.AddInt64(&open, -1) },
					func() { atomic.AddInt64(&open, -1) },
				))
			}),
		)

		done := make(chan struct{})
		started := make(chan struct{})
		go func() {
			defer close(done)

			for i := 0; i < emissions; i++ {
				source.Next(i)

				if i == keys {
					close(started)
				}
			}
		}()

		<-started // unsubscribe while values for existing keys are still in flight
		sub.Unsubscribe()
		<-done

		is.Zero(atomic.LoadInt64(&open), "every emitted group must be completed on teardown")
	}
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

// windowLog records everything a WindowWhen pipeline emits. It is only safe to
// use with sources driven from a single goroutine (e.g. PublishSubject driven
// by the test), which keeps the assertions deterministic and free of sleeps.
type windowLog[T any] struct {
	windows   [][]T
	closed    []bool
	errs      []error
	err       error
	completed bool
}

// observeWindows subscribes to every window as soon as it is emitted.
func observeWindows[T any](source Observable[Observable[T]]) (*windowLog[T], Subscription) {
	log := &windowLog[T]{}

	sub := source.Subscribe(NewObserver(
		func(window Observable[T]) {
			i := len(log.windows)
			log.windows = append(log.windows, []T{})
			log.closed = append(log.closed, false)
			log.errs = append(log.errs, nil)

			window.Subscribe(NewObserver(
				func(value T) { log.windows[i] = append(log.windows[i], value) },
				func(err error) { log.errs[i] = err },
				func() { log.closed[i] = true },
			))
		},
		func(err error) { log.err = err },
		func() { log.completed = true },
	))

	return log, sub
}

func TestOperatorTransformationWindowWhen(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)

	t.Run("collect windows of a synchronous source", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		// Never() boundary: a single window holding every item. Windows must be
		// subscribed while they are open: a window completed before anyone
		// subscribed does not replay its buffered items.
		log, sub := observeWindows(
			Pipe1(
				Just(1, 2, 3),
				WindowWhen[int](Never()),
			),
		)
		defer sub.Unsubscribe()

		is.Equal([][]int{{1, 2, 3}}, log.windows)
		is.Equal([]bool{true}, log.closed)
		is.True(log.completed)
		is.NoError(log.err)
	})

	t.Run("empty source emits one empty window", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		windows, err := Collect(
			Pipe1(
				Empty[int](),
				WindowWhen[int](Never()),
			),
		)
		is.NoError(err)
		is.Len(windows, 1)

		values, err := Collect(windows[0])
		is.NoError(err)
		is.Equal([]int{}, values)
	})

	t.Run("source error propagates without emitting a window", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		windows, err := Collect(
			Pipe1(
				Throw[int](assert.AnError),
				WindowWhen[int](Never()),
			),
		)
		is.EqualError(err, assert.AnError.Error())
		is.Len(windows, 1) // the first window is opened before the source is subscribed

		values, err := Collect(windows[0])
		is.NoError(err) // window is completed, not errored
		is.Equal([]int{}, values)
	})

	t.Run("boundary splits the source into windows", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[string]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		// first window is opened on subscription
		is.Equal([][]int{{}}, log.windows)
		is.Equal([]bool{false}, log.closed)

		source.Next(1)
		source.Next(2)
		is.Equal([][]int{{1, 2}}, log.windows)

		boundary.Next("tick")
		is.Equal([][]int{{1, 2}, {}}, log.windows)
		is.Equal([]bool{true, false}, log.closed)

		source.Next(3)
		boundary.Next("tick")
		source.Next(4)
		source.Next(5)
		is.Equal([][]int{{1, 2}, {3}, {4, 5}}, log.windows)
		is.Equal([]bool{true, true, false}, log.closed)
		is.False(log.completed)

		source.Complete()
		is.Equal([]bool{true, true, true}, log.closed)
		is.True(log.completed)
		is.NoError(log.err)
		is.Equal([]error{nil, nil, nil}, log.errs)
	})

	t.Run("consecutive boundary notifications emit empty windows", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		boundary.Next(0)
		boundary.Next(0)
		source.Next(1)
		source.Complete()

		is.Equal([][]int{{}, {}, {1}}, log.windows)
		is.Equal([]bool{true, true, true}, log.closed)
		is.True(log.completed)
	})

	t.Run("source completion closes the open window and completes", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		source.Next(1)
		source.Complete()

		is.Equal([][]int{{1}}, log.windows)
		is.Equal([]bool{true}, log.closed)
		is.True(log.completed)
		is.NoError(log.err)

		// a late boundary notification must not open a new window
		boundary.Next(0)
		is.Len(log.windows, 1)
	})

	t.Run("source error closes the open window and propagates", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		source.Next(1)
		boundary.Next(0)
		source.Next(2)
		source.Error(assert.AnError)

		is.Equal([][]int{{1}, {2}}, log.windows)
		is.Equal([]bool{true, true}, log.closed)
		is.Equal([]error{nil, nil}, log.errs)
		is.EqualError(log.err, assert.AnError.Error())
		is.False(log.completed)
	})

	t.Run("boundary error closes the open window and propagates", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		source.Next(1)
		boundary.Error(assert.AnError)

		is.Equal([][]int{{1}}, log.windows)
		is.Equal([]bool{true}, log.closed)
		is.EqualError(log.err, assert.AnError.Error())
		is.False(log.completed)
	})

	t.Run("boundary completion closes the open window and completes", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))
		defer sub.Unsubscribe()

		source.Next(1)
		boundary.Complete()

		is.Equal([][]int{{1}}, log.windows)
		is.Equal([]bool{true}, log.closed)
		is.True(log.completed)
		is.NoError(log.err)
	})

	t.Run("early unsubscription releases source and boundary", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		log, sub := observeWindows(WindowWhen[int](boundary.AsObservable())(source))

		source.Next(1)
		is.Equal(1, source.CountObservers())
		is.Equal(1, boundary.CountObservers())

		sub.Unsubscribe()

		is.Equal(0, source.CountObservers())
		is.Equal(0, boundary.CountObservers())

		// nothing is delivered after unsubscription
		source.Next(2)
		boundary.Next(0)
		is.Equal([][]int{{1}}, log.windows)
	})

	t.Run("context is propagated to window items", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		type ctxKey struct{}

		source := NewPublishSubject[int]()
		boundary := NewPublishSubject[int]()

		var gotValue any

		sub := WindowWhen[int](boundary.AsObservable())(source).Subscribe(NewObserver(
			func(window Observable[int]) {
				window.Subscribe(NewObserverWithContext(
					func(ctx context.Context, _ int) { gotValue = ctx.Value(ctxKey{}) },
					func(context.Context, error) {},
					func(context.Context) {},
				))
			},
			func(error) {},
			func() {},
		))
		defer sub.Unsubscribe()

		source.NextWithContext(context.WithValue(context.Background(), ctxKey{}, "hello"), 1)
		is.Equal("hello", gotValue)
	})
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

func TestOperatorTransformationThrottleTimeFirstValueWithLongInterval(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// The monotonic clock counts from process start, so an interval longer than the
	// current uptime must still let the first value through.
	values, err := Collect(
		Pipe1(
			Just(1, 2, 3),
			ThrottleTime[int](time.Hour),
		),
	)
	is.Equal([]int{1}, values)
	is.NoError(err)
}

func TestOperatorTransformationBufferWithCountHugeSize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	chunks, err := Collect(BufferWithCount[int](math.MaxInt)(Just(1, 2, 3)))
	is.Equal([][]int{{1, 2, 3}}, chunks)
	is.NoError(err)
}
