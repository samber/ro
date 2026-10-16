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

//nolint:paralleltest // subtests share the testWithTimeout guard and must not run in parallel
func TestPipe(t *testing.T) {
	testWithTimeout(t, 100*time.Millisecond)

	t.Run("no operator", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			Pipe[int, int](
				Just(1, 2, 3),
			),
		)
		is.Equal([]int{1, 2, 3}, values)
		is.NoError(err)
	})

	t.Run("operators", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
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
	})

	t.Run("error source", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			Pipe[int, int](
				Throw[int](assert.AnError),
			),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("operator input type mismatch", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: *ro.observableImpl[int] does not implements ro.Observable[bool]", func() {
			_, _ = Collect(
				Pipe[int, int](
					Throw[int](assert.AnError),
					passThrough[bool](), // should break here
				),
			)
		})
	})

	t.Run("operator input type mismatch before a later operator", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: *ro.observableImpl[int] does not implements ro.Observable[bool]", func() {
			_, _ = Collect(
				Pipe[int, int](
					Throw[int](assert.AnError),
					passThrough[bool](), // should break here
					passThrough[int](),
				),
			)
		})
	})

	t.Run("non-function operator", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: int is not an operator", func() {
			_, _ = Collect(
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
	})

	t.Run("func without parameters", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: func() is not an operator", func() {
			_, _ = Collect(
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
	})

	t.Run("func without return value", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: func(ro.Observable[int]) is not an operator", func() {
			_, _ = Collect(
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
	})

	t.Run("func without parameters returning an observable", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: func() ro.Observable[int] is not an operator", func() {
			_, _ = Collect(
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
	})

	t.Run("func with a non-observable parameter", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: int does not implements Observable[T]", func() {
			_, _ = Collect(
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
	})

	t.Run("func returning a non-observable", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: string does not implements Observable[T]", func() {
			_, _ = Collect(
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
	})

	t.Run("operator output does not match the next operator input", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: ro.Observable[string] does not implements ro.Observable[int]", func() {
			_, _ = Collect(
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
	})

	t.Run("operator input does not match the previous operator output", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: ro.Observable[int] does not implements ro.Observable[string]", func() {
			values, err := Collect(
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
	})
}

//nolint:paralleltest // subtests share the testWithTimeout guard and must not run in parallel
func TestPipeX(t *testing.T) {
	testWithTimeout(t, 100*time.Millisecond)

	t.Run("Pipe1", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe2", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe3", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe4", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe5", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe6", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe7", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe8", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe9", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe10", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe11", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe12", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe13", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe14", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe15", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe16", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe17", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe18", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe19", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe20", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe21", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe22", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe23", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe24", func(t *testing.T) {
		is := assert.New(t)

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
	})

	t.Run("Pipe25", func(t *testing.T) {
		is := assert.New(t)

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
	})
}

//nolint:paralleltest // subtests share the testWithTimeout guard and must not run in parallel
func TestPipeOp(t *testing.T) {
	testWithTimeout(t, 100*time.Millisecond)

	t.Run("operators", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp[int, int](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
			)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 4}, values)
		is.NoError(err)
	})

	t.Run("error source", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp[int, int](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
			)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("empty source", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp[int, int](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
			)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})

	t.Run("non-function operator", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: int is not an operator", func() {
			_, _ = Collect(
				PipeOp[int, int](
					Map(func(x int) int { return x * 2 }),
					42, // should break here
				)(Just(1, 2, 3)),
			)
		})
	})

	t.Run("func without return value", func(t *testing.T) {
		is := assert.New(t)

		is.PanicsWithError("ro.Pipe: func() is not an operator", func() {
			_, _ = Collect(
				PipeOp[int, int](
					Map(func(x int) int { return x * 2 }),
					func() { panic("never") }, // should break here
				)(Just(1, 2, 3)),
			)
		})
	})
}

//nolint:paralleltest // subtests share the testWithTimeout guard and must not run in parallel
func TestPipeOpX(t *testing.T) {
	testWithTimeout(t, 100*time.Millisecond)

	t.Run("PipeOp1", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp1[int, int](
				Map(func(x int) int { return x * 2 }),
			)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 4, 6}, values)
		is.NoError(err)
	})

	t.Run("PipeOp2", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp2[int, int, int](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
			)(Just(1, 2, 3)),
		)
		is.Equal([]int{2, 4}, values)
		is.NoError(err)
	})

	t.Run("PipeOp3", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp3[int, int, int, string](
				Map(func(x int) int { return x * 2 }),
				Take[int](2),
				Map(strconv.Itoa),
			)(Just(1, 2, 3)),
		)
		is.Equal([]string{"2", "4"}, values)
		is.NoError(err)
	})

	t.Run("PipeOpX with error source", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp1[int, int](
				Map(func(x int) int { return x * 2 }),
			)(Throw[int](assert.AnError)),
		)
		is.Equal([]int{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

	t.Run("PipeOpX with empty source", func(t *testing.T) {
		is := assert.New(t)

		values, err := Collect(
			PipeOp1(
				Map(func(x int) int { return x * 2 }),
			)(Empty[int]()),
		)
		is.Equal([]int{}, values)
		is.NoError(err)
	})
}
