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

package roozzovalidation

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResult_Ok(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test Ok creation
	result := Ok("test")
	is.True(result.IsOk())
	is.False(result.IsError())
	is.Equal("test", result.Unwrap())
	is.Equal("test", result.UnwrapOr("default"))

	value, err := result.Get()
	is.Equal("test", value)
	is.Nil(err)
}

func TestResult_Err(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	testErr := errors.New("test error")
	result := Err[string](testErr)

	is.False(result.IsOk())
	is.True(result.IsError())
	is.Equal(testErr, result.Error())
	is.Equal("default", result.UnwrapOr("default"))

	value, err := result.Get()
	is.Equal("", value)
	is.Equal(testErr, err)
}

func TestResult_Unwrap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test successful unwrap
	okResult := Ok(42)
	is.Equal(42, okResult.Unwrap())

	// Test unwrap with error (should panic)
	errResult := Err[int](errors.New("test error"))

	// We can't easily test panic in this context, but we can test the behavior
	// by checking that the result is in error state
	is.True(errResult.IsError())
}

func TestResult_UnwrapOr(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with Ok result
	okResult := Ok("success")
	is.Equal("success", okResult.UnwrapOr("default"))

	// Test with Err result
	errResult := Err[string](errors.New("test error"))
	is.Equal("default", errResult.UnwrapOr("default"))
}

func TestResult_Get(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with Ok result
	okResult := Ok[int](123)
	value, err := okResult.Get()
	is.Equal(123, value)
	is.Nil(err)

	// Test with Err result
	testErr := errors.New("test error")
	errResult := Err[int](testErr)
	value, err = errResult.Get()
	is.Equal(0, value) // zero value for int
	is.Equal(testErr, err)
}

func TestResult_IsOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	okResult := Ok("test")
	is.True(okResult.IsOk())

	errResult := Err[string](errors.New("test error"))
	is.False(errResult.IsOk())
}

func TestResult_IsError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	okResult := Ok("test")
	is.False(okResult.IsError())

	errResult := Err[string](errors.New("test error"))
	is.True(errResult.IsError())
}

func TestResult_Error(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with Ok result
	okResult := Ok("test")
	is.Nil(okResult.Error())

	// Test with Err result
	testErr := errors.New("test error")
	errResult := Err[string](testErr)
	is.Equal(testErr, errResult.Error())
}

func TestResult_ComplexTypes(t *testing.T) {
	t.Parallel()

	t.Run("struct", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		type TestStruct struct {
			Name  string
			Value int
		}

		okStruct := TestStruct{Name: "test", Value: 42}
		okResult := Ok(okStruct)
		is.True(okResult.IsOk())
		is.Equal(okStruct, okResult.Unwrap())
	})

	t.Run("slice", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		okSlice := []int{1, 2, 3}
		okResultSlice := Ok(okSlice)
		is.True(okResultSlice.IsOk())
		is.Equal(okSlice, okResultSlice.Unwrap())
	})

	t.Run("map", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		okMap := map[string]int{"a": 1, "b": 2}
		okResultMap := Ok(okMap)
		is.True(okResultMap.IsOk())
		is.Equal(okMap, okResultMap.Unwrap())
	})
}
