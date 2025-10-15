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
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorUtilityTap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var nextCount int32

	var errorCount int32

	var completeCount int32

	onNext := func(value int) { atomic.AddInt32(&nextCount, 1) }
	onError := func(error) { atomic.AddInt32(&errorCount, 1) }
	onComplete := func() { atomic.AddInt32(&completeCount, 1) }

	obs := Tap(onNext, onError, onComplete)(Just(1, 2, 3))
	values, err := Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	values, err = Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	is.EqualValues(6, atomic.LoadInt32(&nextCount))
	is.EqualValues(0, atomic.LoadInt32(&errorCount))
	is.EqualValues(2, atomic.LoadInt32(&completeCount))

	values, err = Collect(
		Tap(onNext, onError, onComplete)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
	is.EqualValues(6, atomic.LoadInt32(&nextCount))
	is.EqualValues(0, atomic.LoadInt32(&errorCount))
	is.EqualValues(3, atomic.LoadInt32(&completeCount))

	values, err = Collect(
		Tap(onNext, onError, onComplete)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	is.EqualValues(6, atomic.LoadInt32(&nextCount))
	is.EqualValues(1, atomic.LoadInt32(&errorCount))
	is.EqualValues(3, atomic.LoadInt32(&completeCount))
}

func TestOperatorUtilityTapOnNext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var count int32

	onNext := func(value int) {
		atomic.AddInt32(&count, 1)
	}

	obs := TapOnNext(onNext)(Just(1, 2, 3))
	values, err := Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	values, err = Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	is.EqualValues(6, atomic.LoadInt32(&count))

	values, err = Collect(
		TapOnNext(onNext)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
	is.EqualValues(6, atomic.LoadInt32(&count))

	values, err = Collect(
		TapOnNext(onNext)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	is.EqualValues(6, atomic.LoadInt32(&count))
}

func TestOperatorUtilityTapOnError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var count int32

	onError := func(err error) {
		atomic.AddInt32(&count, 1)
	}

	values, err := Collect(
		TapOnError[int](onError)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	is.EqualValues(0, atomic.LoadInt32(&count))

	values, err = Collect(
		TapOnError[int](onError)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
	is.EqualValues(0, atomic.LoadInt32(&count))

	obs := TapOnError[int](onError)(Throw[int](assert.AnError))
	values, err = Collect(obs)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	values, err = Collect(obs)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	is.EqualValues(2, atomic.LoadInt32(&count))
}

func TestOperatorUtilityTapOnComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var count int32

	onComplete := func() {
		atomic.AddInt32(&count, 1)
	}

	obs := TapOnComplete[int](onComplete)(Just(1, 2, 3))
	values, err := Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	values, err = Collect(obs)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)
	is.EqualValues(2, atomic.LoadInt32(&count))

	values, err = Collect(
		TapOnComplete[int](onComplete)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
	is.EqualValues(3, atomic.LoadInt32(&count))

	values, err = Collect(
		TapOnComplete[int](onComplete)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
	is.EqualValues(3, atomic.LoadInt32(&count))
}

func TestOperatorUtilityTapOnSubscribe(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var count int32

	onSubscribe := func() {
		atomic.AddInt32(&count, 1)
	}

	obs := TapOnSubscribe[int](onSubscribe)(Just(1, 2, 3))
	_, _ = Collect(obs)

	is.EqualValues(1, atomic.LoadInt32(&count))

	_, _ = Collect(obs)
	_, _ = Collect(obs)
	_, _ = Collect(obs)

	is.EqualValues(4, atomic.LoadInt32(&count))

	_, _ = Collect(
		TapOnSubscribe[int](onSubscribe)(Empty[int]()),
	)

	is.EqualValues(5, atomic.LoadInt32(&count))

	_, _ = Collect(
		TapOnSubscribe[int](onSubscribe)(Throw[int](assert.AnError)),
	)

	is.EqualValues(6, atomic.LoadInt32(&count))
}

func TestOperatorUtilityTapOnFinalize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var count int32

	onFinalize := func() {
		atomic.AddInt32(&count, 1)
	}

	obs := TapOnSubscribe[int](onFinalize)(Just(1, 2, 3))
	_, _ = Collect(obs)

	is.EqualValues(1, atomic.LoadInt32(&count))

	_, _ = Collect(obs)
	_, _ = Collect(obs)
	_, _ = Collect(obs)

	is.EqualValues(4, atomic.LoadInt32(&count))

	_, _ = Collect(
		TapOnSubscribe[int](onFinalize)(Empty[int]()),
	)

	is.EqualValues(5, atomic.LoadInt32(&count))

	_, _ = Collect(
		TapOnSubscribe[int](onFinalize)(Throw[int](assert.AnError)),
	)

	is.EqualValues(6, atomic.LoadInt32(&count))
}

func TestOperatorUtilityTimeInterval(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		TimeInterval[int64]()(RangeWithInterval(0, 3, 50*time.Millisecond)),
	)
	expected := []IntervalValue[int64]{
		{Value: 0, Interval: 50 * time.Millisecond},
		{Value: 1, Interval: 50 * time.Millisecond},
		{Value: 2, Interval: 50 * time.Millisecond},
	}
	for i := range expected {
		is.Equal(expected[i].Value, values[i].Value)
		is.InDelta(expected[i].Interval, values[i].Interval, float64(15*time.Millisecond))
	}
	is.Len(values, len(expected))
	is.NoError(err)
}

func TestOperatorUtilityTimestamp(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Timestamp[int64]()(RangeWithInterval(0, 3, 50*time.Millisecond)),
	)
	expected := []TimestampValue[int64]{
		{Value: 0, Timestamp: 50 * time.Millisecond},
		{Value: 1, Timestamp: 100 * time.Millisecond},
		{Value: 2, Timestamp: 150 * time.Millisecond},
	}
	for i := range expected {
		is.Equal(expected[i].Value, values[i].Value)
		is.InDelta(expected[i].Timestamp, values[i].Timestamp, float64(15*time.Millisecond))
	}
	is.Len(values, len(expected))
	is.NoError(err)
}

func TestOperatorUtilityDelay(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// The Complete signal is not delayed.
	values, err := Collect(
		Delay[int](10 * time.Millisecond)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// The Complete signal is send after the delay.
	values, err = Collect(
		Delay[int](100 * time.Millisecond)(
			NewObservable(func(destination Observer[int]) Teardown {
				destination.Next(1)
				destination.Next(2)
				destination.Next(3)
				time.Sleep(100 * time.Millisecond)
				destination.Complete()

				return nil
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// The last message is delayed, but the Complete signal as well.
	values, err = Collect(
		Delay[int](100 * time.Millisecond)(
			NewObservable(func(destination Observer[int]) Teardown {
				destination.Next(1)
				destination.Next(2)
				time.Sleep(100 * time.Millisecond)
				destination.Next(3)
				destination.Complete()

				return nil
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// The last message is delayed, but the Error signal as well.
	values, err = Collect(
		Delay[int](100 * time.Millisecond)(
			NewObservable(func(destination Observer[int]) Teardown {
				destination.Next(1)
				destination.Next(2)
				time.Sleep(150 * time.Millisecond)
				destination.Next(3)
				destination.Error(assert.AnError)

				return nil
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Delay[int](10 * time.Millisecond)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Delay[int](10 * time.Millisecond)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorUtilityRepeatWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		RepeatWith[int64](0)(
			Just[int64](1, 2, 3),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](1)(
			Just[int64](1, 2, 3),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](3)(
			Just[int64](1, 2, 3),
		),
	)
	is.Equal([]int64{1, 2, 3, 1, 2, 3, 1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](0)(
			RangeWithInterval(1, 4, 10*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](1)(
			RangeWithInterval(1, 4, 10*time.Millisecond),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](3)(
			RangeWithInterval(1, 4, 10*time.Millisecond),
		),
	)
	is.Equal([]int64{1, 2, 3, 1, 2, 3, 1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](3)(
			Empty[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		RepeatWith[int64](3)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	is.PanicsWithError(
		"ro.RepeatWith: count must be greater or equal to 0",
		func() {
			values, err = Collect(
				RepeatWith[int64](-1)(
					Just[int64](1, 2, 3),
				),
			)
			is.Equal([]int64{}, values)
			is.NoError(err)
		},
	)
}

func TestOperatorUtilityTimeout(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Timeout[int64](100 * time.Millisecond)(
			RangeWithInterval(1, 4, 10*time.Millisecond),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Timeout[int64](10 * time.Millisecond)(
			RangeWithInterval(1, 4, 100*time.Millisecond),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, "ro.Timeout: timeout after 10ms")

	values, err = Collect(
		Timeout[int64](10 * time.Millisecond)(
			Empty[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Timeout[int64](10 * time.Millisecond)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorUtilityMaterialize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Materialize[int64]()(Range(1, 4)),
	)
	is.Equal([]Notification[int64]{
		{Kind: KindNext, Value: 1, Err: nil},
		{Kind: KindNext, Value: 2, Err: nil},
		{Kind: KindNext, Value: 3, Err: nil},
		{Kind: KindComplete, Value: 0, Err: nil},
	}, values)
	is.NoError(err)

	values, err = Collect(
		Materialize[int64]()(Empty[int64]()),
	)
	is.Equal([]Notification[int64]{
		{Kind: KindComplete, Value: 0, Err: nil},
	}, values)
	is.NoError(err)

	values, err = Collect(
		Materialize[int64]()(NewObservable(func(destination Observer[int64]) Teardown {
			destination.Next(1)
			destination.Next(2)
			destination.Next(3)
			destination.Error(assert.AnError)

			return nil
		})),
	)
	is.Equal([]Notification[int64]{
		{Kind: KindNext, Value: 1, Err: nil},
		{Kind: KindNext, Value: 2, Err: nil},
		{Kind: KindNext, Value: 3, Err: nil},
		{Kind: KindError, Value: 0, Err: assert.AnError},
	}, values)
	is.NoError(err)
	values, err = Collect(
		Materialize[int64]()(Throw[int64](assert.AnError)),
	)
	is.Equal([]Notification[int64]{
		{Kind: KindError, Value: 0, Err: assert.AnError},
	}, values)
	is.NoError(err)
}

func TestOperatorUtilityDematerialize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Dematerialize[int64]()(
			FromSlice(
				[]Notification[int64]{
					{Kind: KindNext, Value: 1, Err: nil},
					{Kind: KindNext, Value: 2, Err: nil},
					{Kind: KindNext, Value: 3, Err: nil},
					{Kind: KindComplete, Value: 0, Err: nil},
				},
			),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Dematerialize[int64]()(
			FromSlice(
				[]Notification[int64]{
					{Kind: KindComplete, Value: 0, Err: nil},
				},
			),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		Dematerialize[int64]()(
			FromSlice(
				[]Notification[int64]{
					{Kind: KindNext, Value: 1, Err: nil},
					{Kind: KindNext, Value: 2, Err: nil},
					{Kind: KindNext, Value: 3, Err: nil},
					{Kind: KindError, Value: 0, Err: assert.AnError},
				},
			),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Dematerialize[int64]()(
			FromSlice(
				[]Notification[int64]{
					{Kind: KindError, Value: 0, Err: assert.AnError},
				},
			),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorSchedulerSubscribeOn(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 400*time.Millisecond)
	is := assert.New(t)

	is.PanicsWithError(
		"ro.SubscribeOn: buffer size must be greater than 0",
		func() {
			_, _ = Collect(
				Pipe2(
					Just[int64](1, 2, 3),
					SubscribeOn[int64](-42),
					Map(func(x int64) int64 {
						time.Sleep(10 * time.Millisecond) // simulate slow processing
						return x
					}),
				),
			)
		},
	)

	values, err := Collect(
		Pipe2(
			Just[int64](1, 2, 3),
			SubscribeOn[int64](42),
			Map(func(x int64) int64 {
				time.Sleep(10 * time.Millisecond) // simulate slow processing
				return x
			}),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	// check that either the upstream or downstream run in a goroutine
	mu := lo.Synchronize()
	order := []int64{}
	values, err = Collect(
		Pipe3(
			Range(1, 4),
			TapOnNext(func(value int64) {
				mu.Do(func() {
					order = append(order, value)
				})
			}),
			SubscribeOn[int64](42),
			TapOnNext(func(value int64) {
				time.Sleep(10 * time.Millisecond)
				mu.Do(func() {
					order = append(order, value*-1)
				})
			}),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.Equal([]int64{1, 2, 3, -1, -2, -3}, order)
	is.NoError(err)

	// check that goroutine is used on downstream instead of upstream
	start := time.Now()
	obs := Pipe1(
		RangeWithInterval(0, 3, 50*time.Millisecond),
		SubscribeOn[int64](42),
	)
	sub := obs.Subscribe(NoopObserver[int64]())

	is.InDelta(150*time.Millisecond, time.Since(start), float64(15*time.Millisecond))
	is.True(sub.IsClosed())

	// @TODO: write some tests for channel buffer overflow
}

func TestOperatorSchedulerObserveOn(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 700*time.Millisecond)
	is := assert.New(t)

	is.PanicsWithError(
		"ro.ObserveOn: buffer size must be greater than 0",
		func() {
			_, _ = Collect(
				Pipe2(
					Just[int64](1, 2, 3),
					ObserveOn[int64](-42),
					Map(func(x int64) int64 {
						time.Sleep(10 * time.Millisecond) // simulate slow processing
						return x
					}),
				),
			)
		},
	)

	values, err := Collect(
		Pipe2(
			Just[int64](1, 2, 3),
			ObserveOn[int64](42),
			Map(func(x int64) int64 {
				time.Sleep(10 * time.Millisecond) // simulate slow processing
				return x
			}),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.NoError(err)

	// check that either the upstream or downstream run in a goroutine
	mu := lo.Synchronize()
	order := []int64{}
	values, err = Collect(
		Pipe3(
			Range(1, 4),
			TapOnNext(func(value int64) {
				mu.Do(func() {
					order = append(order, value)
				})
			}),
			ObserveOn[int64](42),
			TapOnNext(func(value int64) {
				time.Sleep(20 * time.Millisecond)
				mu.Do(func() {
					order = append(order, value*-1)
				})
			}),
		),
	)
	is.Equal([]int64{1, 2, 3}, values)
	is.Equal([]int64{1, 2, 3, -1, -2, -3}, order)
	is.NoError(err)

	// check that goroutine is used on downstream instead of upstream
	start := time.Now()
	obs := Pipe1(
		RangeWithInterval(0, 3, 50*time.Millisecond),
		ObserveOn[int64](42),
	)
	sub := obs.Subscribe(NoopObserver[int64]())

	is.InDelta(0, time.Since(start), float64(15*time.Millisecond))
	is.False(sub.IsClosed())
	sub.Wait() // Note: using .Wait() is not recommended.
	is.InDelta(150*time.Millisecond, time.Since(start), float64(15*time.Millisecond))
	is.True(sub.IsClosed())

	// @TODO: write some tests for channel buffer overflow
}
