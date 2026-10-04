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
	"strconv"
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

// Float steps that are not exactly representable must not change the number of emitted values:
// the range is [start:end), so `end` is never emitted and every value strictly below it is.
func TestOperatorCreationRangeWithStepFloatPrecision(t *testing.T) {
	t.Parallel()
	// The timeout covers the 16 parallel subtests, which wait for the whole package to schedule them.
	testWithTimeout(t, 5*time.Second)

	tests := []struct {
		name       string
		start, end float64
		step       float64
		expected   []float64
	}{
		{"0.1 steps never reach end", 0, 1, 0.1, []float64{0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}},
		{"end is not a multiple of step", 0, 1, 0.3, []float64{0, 0.3, 0.6, 0.9}},
		{"0.3/0.1 is 2.9999999999999996", 0, 0.3, 0.1, []float64{0, 0.1, 0.2}},
		{"0.07/0.01 is 7.000000000000001", 0, 0.07, 0.01, []float64{0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06}},
		{"descending 0.1 steps", 1, 0, 0.1, []float64{1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1}},
		{"descending end is not a multiple of step", 1, 0, 0.3, []float64{1, 0.7, 0.4, 0.1}},
		{"large offset, small step", 1e6, 1000000.02, 0.01, []float64{1e6, 1000000.01}},
		{"step bigger than range", 0, 1, 5, []float64{0}},
	}

	for _, tt := range tests {
		tt := tt // go.mod predates Go 1.22 per-iteration loop variables

		t.Run("RangeWithStep/"+tt.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			values, err := Collect(RangeWithStep(tt.start, tt.end, tt.step))
			is.NoError(err)
			// Len first: InDeltaSlice indexes both slices and panics on a shorter one.
			if is.Len(values, len(tt.expected)) {
				is.InDeltaSlice(tt.expected, values, 1e-9)
			}
		})

		t.Run("RangeWithStepAndInterval/"+tt.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			values, err := Collect(RangeWithStepAndInterval(tt.start, tt.end, tt.step, time.Millisecond))
			is.NoError(err)
			if is.Len(values, len(tt.expected)) {
				is.InDeltaSlice(tt.expected, values, 1e-9)
			}
		})
	}
}

func TestOperatorCreationRangeWithStepEpsilon(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Anchor: with |start|+|end| = 1 and step = 1, the epsilon is the factor times the machine epsilon.
	is.Equal(rangeWithStepEpsilonFactor*float64Epsilon, rangeWithStepEpsilon(0, 1, 1))

	// Both bounds at zero: nothing to absorb.
	is.Equal(0.0, rangeWithStepEpsilon(0, 0, 1))

	// Linear in the magnitude of the bounds.
	is.InEpsilon(10*rangeWithStepEpsilon(0, 1, 1), rangeWithStepEpsilon(0, 10, 1), 1e-12)
	is.InEpsilon(1000*rangeWithStepEpsilon(0, 1, 1), rangeWithStepEpsilon(0, 1000, 1), 1e-12)

	// Inversely proportional to the step.
	is.InEpsilon(2*rangeWithStepEpsilon(0, 1, 0.1), rangeWithStepEpsilon(0, 1, 0.05), 1e-12)

	// Direction and sign of the bounds do not matter, only their magnitude.
	is.Equal(rangeWithStepEpsilon(0, 1, 0.1), rangeWithStepEpsilon(1, 0, 0.1))
	is.Equal(rangeWithStepEpsilon(0, 10, 1), rangeWithStepEpsilon(-5, 5, 1))
	is.Equal(rangeWithStepEpsilon(0, 10, 1), rangeWithStepEpsilon(-10, 0, 1))

	// Tighter than a fixed 1e-9 on small ranges, so a genuine fraction of a step is never swallowed...
	is.Less(rangeWithStepEpsilon(0, 1, 0.1), 1e-9)
	// ...and looser on large offsets, where the subtraction error exceeds 1e-9.
	is.Greater(rangeWithStepEpsilon(1e6, 1e6+1, 0.01), 1e-9)
	is.Greater(rangeWithStepEpsilon(1e9, 1e9+1, 0.1), 1e-9)

	// Stays far below one step for every realistic input, so it can never drop a whole value.
	is.Less(rangeWithStepEpsilon(1e9, 1e9+1, 0.1), 0.5)
}

func TestOperatorCreationRangeWithStepCount(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name       string
		start, end float64
		step       float64
		expected   int64
	}{
		{"exact multiple of an exact step", 0, 1, 0.5, 2},
		{"0.1 steps never reach end", 0, 1, 0.1, 10},
		{"end is not a multiple of step", 0, 1, 0.3, 4},
		{"0.3/0.1 is 2.9999999999999996", 0, 0.3, 0.1, 3},
		{"0.07/0.01 is 7.000000000000001", 0, 0.07, 0.01, 7},
		{"1.1/0.1 emits 11 values", 0, 1.1, 0.1, 11},
		{"descending 0.1 steps", 1, 0, 0.1, 10},
		{"descending end is not a multiple of step", 1, 0, 0.3, 4},
		{"negative bounds, ascending", -1, 0, 0.1, 10},
		{"negative bounds, descending", -0.3, -0.6, 0.1, 3},
		{"bounds around zero", -0.5, 0.5, 0.25, 4},
		{"large offset, small step", 1e6, 1000000.02, 0.01, 2},
		{"huge offset, one step", 1e9, 1000000000.1, 0.1, 1},
		// A fraction of a step above a multiple is a genuine extra value: 1 < 1.000000001.
		{"end just above a multiple of step", 0, 1 + 1e-9, 1, 2},
		{"end just below a multiple of step", 0, 1 - 1e-9, 1, 1},
		// start != end, so the range always holds at least start.
		{"step equal to range", 0, 1, 1, 1},
		{"step bigger than range", 0, 1, 5, 1},
		{"tiny range", 0, 1e-300, 1, 1},
		// The epsilon (~1.8e-6) exceeds the quotient (~1e-6): ceil would give 0 without the clamp.
		{"gap smaller than the epsilon at a large offset", 1e9, 1e9 + 1e-6, 1, 1},
		{"gap smaller than the epsilon at a large offset, descending", 1e9 + 1e-6, 1e9, 1, 1},
		{"tiny range, descending", 0, -1e-300, 1, 1},
	}

	for _, tt := range tests {
		tt := tt // go.mod predates Go 1.22 per-iteration loop variables

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, tt.expected, rangeWithStepCount(tt.start, tt.end, tt.step))
		})
	}
}

// The count does not depend on the walking direction.
func TestOperatorCreationRangeWithStepCountSymmetry(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	for _, step := range []float64{0.1, 0.3, 0.7, 1, 2.5} {
		for _, bound := range []float64{0.07, 0.3, 1, 1.1, 5, 42.5} {
			is.Equal(rangeWithStepCount(0, bound, step), rangeWithStepCount(bound, 0, step), "step=%v bound=%v", step, bound)
			is.Equal(rangeWithStepCount(-bound, bound, step), rangeWithStepCount(bound, -bound, step), "step=%v bound=%v", step, bound)
		}
	}
}

// For an end written as the decimal literal of offset+k*step, the count is exactly k.
func TestOperatorCreationRangeWithStepCountDecimalLiterals(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	const maxK = 200

	for _, step := range []float64{0.1, 0.01, 0.001, 0.05, 0.2, 0.3, 0.7} {
		for _, offset := range []float64{0, -3, 1e3, 1e6} {
			for k := 1; k <= maxK; k++ {
				// Round-trip through a decimal string to get the literal a caller would type.
				end, err := strconv.ParseFloat(strconv.FormatFloat(offset+float64(k)*step, 'f', 6, 64), 64)
				is.NoError(err)

				if !is.Equal(int64(k), rangeWithStepCount(offset, end, step), "offset=%v step=%v k=%d end=%v", offset, step, k, end) {
					return
				}
			}
		}
	}
}

func TestOperatorCreationRangeWithStepValue(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// The first value is start itself, whatever the direction and the step.
	is.Equal(0.0, rangeWithStepValue(0, 0.1, 1, 0))
	is.Equal(0.0, rangeWithStepValue(0, 0.1, -1, 0))
	is.Equal(1.0, rangeWithStepValue(1, 0.1, -1, 0))
	is.Equal(-2.5, rangeWithStepValue(-2.5, 7, 1, 0))
	is.Equal(1e9, rangeWithStepValue(1e9, 0.1, 1, 0))

	// Ascending and descending.
	is.InDelta(0.3, rangeWithStepValue(0, 0.1, 1, 3), 1e-12)
	is.InDelta(0.7, rangeWithStepValue(1, 0.1, -1, 3), 1e-12)
	is.InDelta(0.9, rangeWithStepValue(0, 0.3, 1, 3), 1e-12)

	// Negative start.
	is.InDelta(0.0, rangeWithStepValue(-1, 0.5, 1, 2), 1e-12)
	is.InDelta(-1.5, rangeWithStepValue(-1, 0.5, -1, 1), 1e-12)

	// Step bigger than any range stays exact.
	is.Equal(5.0, rangeWithStepValue(0, 5, 1, 1))
	is.Equal(-5.0, rangeWithStepValue(0, 5, -1, 1))
}

// Computing start + i*step keeps the error independent from i, whereas accumulating the step does not.
func TestOperatorCreationRangeWithStepValueNoDrift(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	const (
		iterations = 1_000_000
		step       = 0.1
		tolerance  = 1e-9
	)

	accumulated := 0.0
	for i := 0; i < iterations; i++ {
		accumulated += step
	}

	// Guards the premise: the naive sum has drifted beyond the tolerance.
	is.Greater(math.Abs(accumulated-iterations*step), tolerance)

	is.InDelta(iterations*step, rangeWithStepValue(0, step, 1, iterations), tolerance)
	is.InDelta(-iterations*step, rangeWithStepValue(0, step, -1, iterations), tolerance)
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

func TestOperatorCreationRangeExtremes(t *testing.T) {
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
