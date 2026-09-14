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

	"github.com/stretchr/testify/assert"
)

func TestPipe(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe[int, int](
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe[int, int](
			Just(1, 2, 3),
			Map(func(x int) int {
				return x * 2
			}),
			Take[int](2),
		),
	)
	is.Equal([]int{2, 4}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe[int, int](
			Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	is.PanicsWithError("ro.Pipe: *ro.observableImpl[int] does not implements ro.Observable[bool]", func() {
		values, err = Collect(
			Pipe[int, int](
				Throw[int](assert.AnError),
				passThrough[bool](), // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: *ro.observableImpl[int] does not implements ro.Observable[bool]", func() {
		values, err = Collect(
			Pipe[int, int](
				Throw[int](assert.AnError),
				passThrough[bool](), // should break here
				passThrough[int](),
			),
		)
	})

	is.PanicsWithError("ro.Pipe: int is not an operator", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				42, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: func() is not an operator", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				func() {
					panic("never")
				}, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: func(ro.Observable[int]) is not an operator", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				func(Observable[int]) {
					panic("never")
				}, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: func() ro.Observable[int] is not an operator", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				func() Observable[int] {
					panic("never")
				}, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: int does not implements Observable[T]", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				func(int) Observable[int] {
					panic("never")
				}, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: string does not implements Observable[T]", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				func(Observable[int]) string {
					panic("never")
				}, // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: ro.Observable[string] does not implements ro.Observable[int]", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				Map(strconv.Itoa), // should break here
			),
		)
	})

	is.PanicsWithError("ro.Pipe: ro.Observable[int] does not implements ro.Observable[string]", func() {
		values, err = Collect(
			Pipe[int, int](
				Just(1, 2, 3),
				Map(func(x int) int {
					return x * 2
				}),
				Take[int](2),
				Map(func(x string) int {
					return 42
				}), // should break here
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestPipeX(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// Pipe1
	{
		values, err := Collect(
			Pipe1(
				Just(1, 2, 3),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe1(
				Throw[int](assert.AnError),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe2
	{
		values, err := Collect(
			Pipe2(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe2(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe3
	{
		values, err := Collect(
			Pipe3(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe3(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe4
	{
		values, err := Collect(
			Pipe4(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe4(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe5
	{
		values, err := Collect(
			Pipe5(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe5(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe6
	{
		values, err := Collect(
			Pipe6(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe6(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe7
	{
		values, err := Collect(
			Pipe7(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe7(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe8
	{
		values, err := Collect(
			Pipe8(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe8(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe9
	{
		values, err := Collect(
			Pipe9(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe9(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe10
	{
		values, err := Collect(
			Pipe10(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe10(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe11
	{
		values, err := Collect(
			Pipe11(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe11(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe12
	{
		values, err := Collect(
			Pipe12(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe12(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe13
	{
		values, err := Collect(
			Pipe13(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe13(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe14
	{
		values, err := Collect(
			Pipe14(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe14(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe15
	{
		values, err := Collect(
			Pipe15(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe15(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe16
	{
		values, err := Collect(
			Pipe16(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe16(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe17
	{
		values, err := Collect(
			Pipe17(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe17(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe18
	{
		values, err := Collect(
			Pipe18(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe18(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe19
	{
		values, err := Collect(
			Pipe19(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe19(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe20
	{
		values, err := Collect(
			Pipe20(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe20(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe21
	{
		values, err := Collect(
			Pipe21(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe21(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe22
	{
		values, err := Collect(
			Pipe22(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe22(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe23
	{
		values, err := Collect(
			Pipe23(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe23(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe24
	{
		values, err := Collect(
			Pipe24(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe24(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}

	// Pipe25
	{
		values, err := Collect(
			Pipe25(
				Just(1, 2, 3),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)

		values, err = Collect(
			Pipe25(
				Throw[int](assert.AnError),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
				passThrough[int](),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	}
}

func TestPipeOp(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		PipeOp[int, int](
			Map(func(x int) int { return x * 2 }),
			Take[int](2),
		)(Just(1, 2, 3)),
	)
	is.Equal([]int{2, 4}, values)
	is.NoError(err)

	values, err = Collect(
		PipeOp[int, int](
			Map(func(x int) int { return x * 2 }),
			Take[int](2),
		)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		PipeOp[int, int](
			Map(func(x int) int { return x * 2 }),
			Take[int](2),
		)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	is.PanicsWithError("ro.Pipe: int is not an operator", func() {
		_, _ = Collect(
			PipeOp[int, int](
				Map(func(x int) int { return x * 2 }),
				42, // should break here
			)(Just(1, 2, 3)),
		)
	})

	is.PanicsWithError("ro.Pipe: func() is not an operator", func() {
		_, _ = Collect(
			PipeOp[int, int](
				Map(func(x int) int { return x * 2 }),
				func() { panic("never") }, // should break here
			)(Just(1, 2, 3)),
		)
	})
}

func TestPipeOpX(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	// PipeOp1
	{
		values, err := Collect(
			PipeOp1[int, int](
				Map(func(x int) int { return x * 2 }),
			)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 4, 6}, values)
		is.NoError(err)
	}

	// PipeOp2
	{
		values, err := Collect(
			PipeOp2[int, int, int](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
			)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 4}, values)
		is.NoError(err)
	}

	// PipeOp3
	{
		values, err := Collect(
			PipeOp3[int, int, int, string](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
				Map(strconv.Itoa),
			)(Just(1, 2, 3)),
		)
		is.Equal([]string{"2", "4"}, values)
		is.NoError(err)
	}

	// PipeOpX with error source
	values, err := Collect(
		PipeOp1[int, int](
			Map(func(x int) int { return x * 2 }),
		)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	// PipeOpX with empty source
	values, err = Collect(
		PipeOp1(
			Map(func(x int) int { return x * 2 }),
		)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)
}
