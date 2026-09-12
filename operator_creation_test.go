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
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorCreationOf(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Of(1, 2, 3),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Of[int](),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestOperatorCreationJust(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Just(1, 2, 3),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Just[int](),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestOperatorCreationStart(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Start(func() int {
			return 42
		}),
	)
	is.Equal([]int{42}, values)
	is.NoError(err)
}

func TestOperatorCreationTimer(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	start := time.Now()

	values, err := Collect(
		Timer(50 * time.Millisecond),
	)
	is.Equal([]time.Duration{50 * time.Millisecond}, values)
	is.NoError(err)
	is.InDelta(50*time.Millisecond, time.Since(start), float64(10*time.Millisecond))
}

func TestOperatorCreationInterval(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 400*time.Millisecond)
	is := assert.New(t)

	interval := 50 * time.Millisecond

	sync := lo.Synchronize()
	output := []IntervalValue[int64]{}

	sub := Pipe1(
		Interval(interval),
		TimeInterval[int64](),
	).Subscribe(
		NewObserver(
			func(v IntervalValue[int64]) {
				sync.Do(func() {
					output = append(output, v)
				})
			},
			func(err error) {
				is.Fail("never")
			},
			func() {
				is.Fail("never")
			},
		),
	)

	time.Sleep(175 * time.Millisecond)

	is.False(sub.IsClosed())
	sub.Unsubscribe()
	is.True(sub.IsClosed())

	expected := []IntervalValue[int64]{
		{Value: 0, Interval: interval},
		{Value: 1, Interval: interval},
		{Value: 2, Interval: interval},
	}
	sync.Do(func() {
		is.Len(output, 3)
		for i := 0; i < 3; i++ {
			is.Equal(expected[i].Value, output[i].Value)
			is.InDelta(expected[i].Interval, output[i].Interval, float64(15*time.Millisecond))
		}
	})
}

func TestOperatorCreationIntervalWithInitial(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 400*time.Millisecond)
	is := assert.New(t)

	interval := 50 * time.Millisecond

	sync := lo.Synchronize()
	output := []IntervalValue[int64]{}

	sub := Pipe1(
		IntervalWithInitial(interval*2, interval),
		TimeInterval[int64](),
	).Subscribe(
		NewObserver(
			func(v IntervalValue[int64]) {
				sync.Do(func() {
					output = append(output, v)
				})
			},
			func(err error) {
				is.Fail("never")
			},
			func() {
				is.Fail("never")
			},
		),
	)

	time.Sleep(225 * time.Millisecond)

	is.False(sub.IsClosed())
	sub.Unsubscribe()
	is.True(sub.IsClosed())

	expected := []IntervalValue[int64]{
		{Value: 0, Interval: interval * 2},
		{Value: 1, Interval: interval},
		{Value: 2, Interval: interval},
	}
	sync.Do(func() {
		is.Len(output, 3)
		for i := 0; i < 3; i++ {
			is.Equal(expected[i].Value, output[i].Value)
			is.InDelta(expected[i].Interval, output[i].Interval, float64(15*time.Millisecond))
		}
	})
}

func TestOperatorCreationIntervalWithZeroInitial(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, time.Second)
	is := assert.New(t)

	values := make(chan int64, 1)
	sub := IntervalWithInitial(0, time.Hour).Subscribe(OnNext(func(value int64) {
		values <- value
	}))
	defer sub.Unsubscribe()

	select {
	case value := <-values:
		is.Equal(int64(0), value)
	default:
		t.Fatal("initial value must be emitted before Subscribe returns")
	}
	sub.Unsubscribe()
	is.True(sub.IsClosed())

	output, err := Collect(Take[int64](1)(IntervalWithInitial(0, time.Hour)))
	is.NoError(err)
	is.Equal([]int64{0}, output)

	output, err = Collect(Take[int64](3)(IntervalWithInitial(0, 10*time.Millisecond)))
	is.NoError(err)
	is.Equal([]int64{0, 1, 2}, output)
}

func TestOperatorCreationIntervalWithZeroInitialCancellation(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, time.Second)
	is := assert.New(t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	values := []int64{}
	var emittedCtx context.Context
	sub := IntervalWithInitial(0, time.Hour).SubscribeWithContext(ctx, OnNextWithContext(func(actualCtx context.Context, value int64) {
		emittedCtx = actualCtx
		values = append(values, value)
		cancel()
	}))
	defer sub.Unsubscribe()
	sub.Wait()

	is.Equal([]int64{0}, values)
	is.Equal(ctx, emittedCtx)
	is.True(sub.IsClosed())
}

func TestOperatorCreationRange(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Range(1, 5),
	)
	is.Equal([]int64{1, 2, 3, 4}, values)
	is.NoError(err)

	values, err = Collect(
		Range(5, 5),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Range(5, 1),
	)
	is.Equal([]int64{5, 4, 3, 2}, values)
	is.NoError(err)
}

func TestOperatorCreationRangeWithStep(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	is.PanicsWithError("ro.RangeWithStep: step must be greater than 0", func() {
		RangeWithStep(1, 5, 0)
	})

	is.PanicsWithError("ro.RangeWithStep: step must be greater than 0", func() {
		RangeWithStep(1, 5, -42)
	})

	values, err := Collect(
		RangeWithStep(1, 5, 0.5),
	)
	is.Equal([]float64{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithStep(5, 5, 0.5),
	)
	is.Equal([]float64{}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithStep(5, 1, 0.5),
	)
	is.Equal([]float64{5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5}, values)
	is.NoError(err)
}

func TestOperatorCreationRangeWithInterval(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	interval := 50 * time.Millisecond

	// duration testing — verify that values are emitted at the expected interval
	intervalValues, err := Collect(
		Pipe1(
			RangeWithInterval(1, 4, interval),
			TimeInterval[int64](),
		),
	)
	is.Len(intervalValues, 3)
	is.NoError(err)
	for i := 0; i < 3; i++ {
		is.Equal(int64(1+i), intervalValues[i].Value)
		is.InDelta(interval, intervalValues[i].Interval, float64(25*time.Millisecond))
	}

	// value testing
	values, err := Collect(
		RangeWithInterval(1, 5, 10*time.Millisecond),
	)
	is.Equal([]int64{1, 2, 3, 4}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithInterval(5, 5, 10*time.Millisecond),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithInterval(6, 2, 10*time.Millisecond),
	)
	is.Equal([]int64{6, 5, 4, 3}, values)
	is.NoError(err)
}

func TestOperatorCreationRangeWithStepAndInterval(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	interval := 50 * time.Millisecond

	// duration testing — verify that values are emitted at the expected interval
	intervalValues, err := Collect(
		Pipe1(
			RangeWithStepAndInterval(1, 4, 0.5, interval),
			TimeInterval[float64](),
		),
	)
	is.Len(intervalValues, 6) // [1, 4) with step 0.5 = 6 values
	is.NoError(err)
	for i := 0; i < 6; i++ {
		is.InDelta(interval, intervalValues[i].Interval, float64(25*time.Millisecond))
	}

	// panics
	is.PanicsWithError("ro.RangeWithStepAndInterval: step must be greater than 0", func() {
		RangeWithStepAndInterval(1, 5, 0, 10*time.Millisecond)
	})

	is.PanicsWithError("ro.RangeWithStepAndInterval: step must be greater than 0", func() {
		RangeWithStepAndInterval(1, 5, -42, 10*time.Millisecond)
	})

	// value testing
	values, err := Collect(
		RangeWithStepAndInterval(1, 5, 0.5, 10*time.Millisecond),
	)
	is.Equal([]float64{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithStepAndInterval(5, 5, 0.5, 10*time.Millisecond),
	)
	is.Equal([]float64{}, values)
	is.NoError(err)

	values, err = Collect(
		RangeWithStepAndInterval(6, 2, 0.5, 10*time.Millisecond),
	)
	is.Equal([]float64{6, 5.5, 5, 4.5, 4, 3.5, 3, 2.5}, values)
	is.NoError(err)
}

func TestOperatorCreationRepeat(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values1, err := Collect(
		Repeat(1, 3),
	)
	is.Equal([]int{1, 1, 1}, values1)
	is.NoError(err)

	values2, err := Collect(
		Repeat("foobar", 3),
	)
	is.Equal([]string{"foobar", "foobar", "foobar"}, values2)
	is.NoError(err)

	values3, err := Collect(
		Repeat(assert.AnError, 3),
	)
	is.Equal([]error{assert.AnError, assert.AnError, assert.AnError}, values3)
	is.NoError(err)

	values2, err = Collect(
		Repeat("foobar", 0),
	)
	is.Equal([]string{}, values2)
	is.NoError(err)

	is.PanicsWithError("ro.Repeat: count must be greater or equal to 0", func() {
		Repeat("foobar", -42)
	})
}

func TestOperatorCreationRepeatWithInterval(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	interval := 50 * time.Millisecond

	// basic values
	values, err := Collect(
		RepeatWithInterval(1, 3, interval),
	)
	is.Equal([]int{1, 1, 1}, values)
	is.NoError(err)

	// empty
	stringValues, err := Collect(
		RepeatWithInterval("foobar", 0, interval),
	)
	is.Equal([]string{}, stringValues)
	is.NoError(err)

	// error values
	errorValues, err := Collect(
		RepeatWithInterval(assert.AnError, 3, interval),
	)
	is.Equal([]error{assert.AnError, assert.AnError, assert.AnError}, errorValues)
	is.NoError(err)

	// panic on negative count
	is.PanicsWithError("ro.RepeatWithInterval: count must be greater or equal to 0", func() {
		RepeatWithInterval(1, -1, interval)
	})

	// duration testing
	intervalValues, err := Collect(
		Pipe1(
			RepeatWithInterval(0, 3, interval),
			TimeInterval[int](),
		),
	)
	is.Len(intervalValues, 3)
	is.NoError(err)
	for i := 0; i < 3; i++ {
		is.InDelta(interval, intervalValues[i].Interval, float64(25*time.Millisecond))
	}
}

func TestOperatorCreationFromChannel(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	ch := make(chan int, 5)
	ch <- 1
	ch <- 2
	ch <- 3

	close(ch)

	// normal case
	values, err := Collect(
		FromChannel(ch),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// already closed
	values, err = Collect(
		FromChannel(ch),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	// Late closing
	start := time.Now()

	ch = make(chan int, 5)
	ch <- 1
	ch <- 2
	ch <- 3

	go func() {
		time.Sleep(50 * time.Millisecond)
		close(ch)
	}()

	values, err = Collect(
		FromChannel(ch),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	is.InDelta(50*time.Millisecond, time.Since(start), float64(10*time.Millisecond))

	// nil channel
	values, err = Collect(
		FromChannel(ch),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	// early unsubscription
	ch = make(chan int, 5)

	go func() {
		time.Sleep(25 * time.Millisecond)

		ch <- 1
		ch <- 2
		ch <- 3

		time.Sleep(50 * time.Millisecond)

		ch <- 4

		close(ch)
	}()

	sync := lo.Synchronize()
	output := []int{}

	var sub Subscription

	sub = FromChannel(ch).
		Subscribe(
			NewObserver(
				func(v int) {
					sync.Do(func() {
						output = append(output, v)
					})
					sub.Unsubscribe()
				},
				func(err error) {
					is.Fail("never")
				},
				func() {
					is.Fail("never")
				},
			),
		)

	is.False(sub.IsClosed())
	sync.Do(func() {
		is.Equal([]int{}, output)
	})

	time.Sleep(50 * time.Millisecond)

	sub.Unsubscribe()
	is.True(sub.IsClosed())
	sync.Do(func() {
		is.Equal([]int{1}, output)
	})
}

func TestOperatorCreationFromSlice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		FromSlice([]int{1, 2, 3}),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		FromSlice([]int{1, 2, 3}, []int{4, 5, 6}),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		FromSlice([]int{}),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestOperatorCreationEmpty(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Empty[int](),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}

func TestOperatorCreationNever(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	done := uint32(0)

	sub := Never().Subscribe(
		NewObserver(
			func(v struct{}) {
				is.Fail("never")
			},
			func(err error) {
				is.Fail("never")
			},
			func() {
				isDone := atomic.LoadUint32(&done)
				if isDone == 0 {
					is.Fail("never")
				} else {
					is.Equal(1, isDone)
				}
			},
		),
	)

	time.AfterFunc(50*time.Millisecond, func() {
		is.False(sub.IsClosed())
		is.Equal(uint32(0), atomic.LoadUint32(&done))
		atomic.CompareAndSwapUint32(&done, 0, 1)
		sub.Unsubscribe()
		is.True(sub.IsClosed())
	})

	is.False(sub.IsClosed())
}

func TestOperatorCreationThrown(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Throw[int](nil),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Throw[int](assert.AnError),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationDefer(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Defer(func() Observable[int] {
			return Of(1, 2, 3)
		}),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Defer(func() Observable[int] {
			return Throw[int](assert.AnError)
		}),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationFuture(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 300*time.Millisecond)
	is := assert.New(t)

	start := time.Now()

	values, err := Collect(
		Future(func() (int, error) {
			time.Sleep(100 * time.Millisecond)
			return 42, nil
		}),
	)
	is.Equal([]int{42}, values)
	is.NoError(err)
	is.InDelta(100*time.Millisecond, time.Since(start), float64(20*time.Millisecond))

	values, err = Collect(
		Future(func() (int, error) {
			time.Sleep(100 * time.Millisecond)
			return 42, assert.AnError
		}),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	is.InDelta(200*time.Millisecond, time.Since(start), float64(40*time.Millisecond))
}

func TestOperatorCreationMerge(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// sequential
	values, err := Collect(
		Merge(
			RangeWithInterval(6, 9, 100*time.Millisecond), // third
			RangeWithInterval(3, 6, 10*time.Millisecond),  // second
			Just[int64](0, 1, 2),                          // first
		),
	)
	is.Equal([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8}, values)
	is.NoError(err)

	// parallel
	values, err = Collect(
		Merge(
			RangeWithInterval(0, 3, 50*time.Millisecond),
			RangeWithInterval(0, 3, 50*time.Millisecond),
			RangeWithInterval(0, 3, 50*time.Millisecond),
		),
	)
	is.Equal([]int64{0, 0, 0, 1, 1, 1, 2, 2, 2}, values)
	is.NoError(err)

	// concurrent
	values, err = Collect(
		Merge(
			RangeWithInterval(0, 3, 60*time.Millisecond),
			Delay[int64](20*time.Millisecond)(RangeWithInterval(3, 6, 60*time.Millisecond)),
			Delay[int64](40*time.Millisecond)(RangeWithInterval(6, 9, 60*time.Millisecond)),
		),
	)
	is.Equal([]int64{0, 3, 6, 1, 4, 7, 2, 5, 8}, values)
	is.NoError(err)

	values, err = Collect(
		Merge[int64](),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Merge(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationCombineLatest2(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		CombineLatest2(
			Of(1),
			Of(2),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{lo.T2(1, 2)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		CombineLatest2(
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		CombineLatest2(
			Throw[int](assert.AnError),
			Of(2),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationCombineLatest3(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		CombineLatest3(
			Of(1),
			Of(2),
			Of(3),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{lo.T3(1, 2, 3)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		CombineLatest3(
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		CombineLatest3(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationCombineLatest4(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		CombineLatest4(
			Of(1),
			Of(2),
			Of(3),
			Of(4),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{lo.T4(1, 2, 3, 4)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		CombineLatest4(
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		CombineLatest4(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
			Of(4),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationCombineLatest5(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		CombineLatest5(
			Of(1),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{lo.T5(1, 2, 3, 4, 5)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		CombineLatest5(
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		CombineLatest5(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationCombineLatestAny(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		CombineLatestAny(
			Of[any](1),
			Of[any]("foo"),
		),
	)
	is.Equal([][]any{{1, "foo"}}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		CombineLatestAny(
			Empty[any](),
			Empty[any](),
		),
	)
	is.Equal([][]any{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		CombineLatestAny(
			Throw[any](assert.AnError),
			Of[any]("foo"),
		),
	)
	is.Equal([][]any{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip(
			Of(1, 2),
			Of(3, 4),
		),
	)
	is.Equal([][]int{{1, 3}, {2, 4}}, values)
	is.NoError(err)

	// empty source
	values, err = Collect(
		Zip(
			Empty[int](),
			Of(1),
		),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	// single value per source
	values, err = Collect(
		Zip(
			Of(1),
			Of(2),
		),
	)
	is.Equal([][]int{{1, 2}}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip(
			Throw[int](assert.AnError),
			Of(1),
		),
	)
	is.Equal([][]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip2(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip2(
			Of(1),
			Of(2),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{lo.T2(1, 2)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		Zip2(
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip2(
			Throw[int](assert.AnError),
			Of(2),
		),
	)
	is.Equal([]lo.Tuple2[int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip3(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip3(
			Of(1),
			Of(2),
			Of(3),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{lo.T3(1, 2, 3)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		Zip3(
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip3(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
		),
	)
	is.Equal([]lo.Tuple3[int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip4(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip4(
			Of(1),
			Of(2),
			Of(3),
			Of(4),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{lo.T4(1, 2, 3, 4)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		Zip4(
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip4(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
			Of(4),
		),
	)
	is.Equal([]lo.Tuple4[int, int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip5(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip5(
			Of(1),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{lo.T5(1, 2, 3, 4, 5)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		Zip5(
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip5(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
		),
	)
	is.Equal([]lo.Tuple5[int, int, int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationZip6(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// basic case
	values, err := Collect(
		Zip6(
			Of(1),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
			Of(6),
		),
	)
	is.Equal([]lo.Tuple6[int, int, int, int, int, int]{lo.T6(1, 2, 3, 4, 5, 6)}, values)
	is.NoError(err)

	// empty sources
	values, err = Collect(
		Zip6(
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
			Empty[int](),
		),
	)
	is.Equal([]lo.Tuple6[int, int, int, int, int, int]{}, values)
	is.NoError(err)

	// error propagation
	values, err = Collect(
		Zip6(
			Throw[int](assert.AnError),
			Of(2),
			Of(3),
			Of(4),
			Of(5),
			Of(6),
		),
	)
	is.Equal([]lo.Tuple6[int, int, int, int, int, int]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationConcat(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Concat(
			Just(1, 2, 3),
			Just(4, 5, 6),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Concat(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Concat(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationRace(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	// First source to emit wins
	values, err := Collect(
		Race(
			RangeWithInterval(0, 2, 100*time.Millisecond),
			Just[int64](99),
		),
	)
	is.Equal([]int64{99}, values)
	is.NoError(err)

	values, err = Collect(
		Race(
			Just[int64](1, 2, 3),
			Just[int64](4, 5, 6),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	// Empty source
	values, err = Collect(
		Race(Empty[int64]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	// Error source
	values, err = Collect(
		Race(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationAmb(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	// Amb is an alias for Race — first source to emit wins
	values, err := Collect(
		Amb(
			RangeWithInterval(0, 2, 100*time.Millisecond),
			Just[int64](99),
		),
	)
	is.Equal([]int64{99}, values)
	is.NoError(err)

	values, err = Collect(
		Amb(
			Just[int64](1, 2, 3),
			Just[int64](4, 5, 6),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Amb(Empty[int64]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Amb(Throw[int64](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCreationRandIntN(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		RandIntN(10, 3),
	)

	is.Len(values, 3)

	for _, v := range values {
		is.True(v >= 0 && v < 10)
	}

	is.NoError(err)
}

func TestOperatorCreationRandFloat64(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		RandFloat64(3),
	)
	is.Len(values, 3)

	for _, v := range values {
		is.True(v >= 0 && v < 1)
	}

	is.NoError(err)
}
