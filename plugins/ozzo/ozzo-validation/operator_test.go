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
	"context"
	"regexp"
	"testing"

	ozzo "github.com/go-ozzo/ozzo-validation/v4"
	"github.com/go-ozzo/ozzo-validation/v4/is"
	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

// Test data structures
type User struct {
	Name  string `json:"name"`
	Email string `json:"email"`
	Age   int    `json:"age"`
}

type ValidatableUser struct {
	Name  string `json:"name"`
	Email string `json:"email"`
	Age   int    `json:"age"`
}

func (u ValidatableUser) Validate() error {
	return ozzo.ValidateStruct(&u,
		ozzo.Field(&u.Name, ozzo.Required, ozzo.Length(1, 50)),
		ozzo.Field(&u.Email, ozzo.Required, is.Email),
		ozzo.Field(&u.Age, ozzo.Required, ozzo.Min(18)),
	)
}

func (u ValidatableUser) ValidateWithContext(ctx context.Context) error {
	return ozzo.ValidateStructWithContext(ctx, &u,
		ozzo.Field(&u.Name, ozzo.Required, ozzo.Length(1, 50)),
		ozzo.Field(&u.Email, ozzo.Required, is.Email),
		ozzo.Field(&u.Age, ozzo.Required, ozzo.Min(18)),
	)
}

func TestValidate(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test successful validation with simple string
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("test"),
			Validate[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())

	// Test failed validation with simple string
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just(""),
			Validate[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsError())
	is.NotNil(values[0].Error())

	// Test empty observable
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Empty[string](),
			Validate[string](ozzo.Required),
		),
	)
	is.Nil(err)
	is.Empty(values)

	// Test error observable
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			Validate[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidate_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	// Test with URL validation
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("https://example.com"),
			Validate[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("https://example.com", values[0].Unwrap())

	// Test with invalid URL
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("not-a-url"),
			Validate[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())

	// Test with numeric validation
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("123"),
			Validate[string](ozzo.Required, ozzo.Match(regexp.MustCompile(`^[0-9]+$`))),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("123", values[0].Unwrap())

	// Test with invalid number
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("abc"),
			Validate[string](ozzo.Required, ozzo.Match(regexp.MustCompile(`^[0-9]+$`))),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())
}

func TestValidate_complexValidation(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	// Test with complex validation rules
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("valid@email.com"),
			Validate[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("valid@email.com", values[0].Unwrap())

	// Test with complex validation rules that fail
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("invalid-email"),
			Validate[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())
}

func TestValidate_edgeCases(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with nil rules
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("test"),
			Validate[string](),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())

	// Test with multiple rules
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("test"),
			Validate[string](ozzo.Required, ozzo.Length(1, 10), ozzo.Match(regexp.MustCompile(`^[a-z]+$`))),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())
}

func TestValidate_simpleTypes(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with string validation
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("test"),
			Validate[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())

	// Test with int validation
	intValues, err := ro.Collect(
		ro.Pipe1(
			ro.Just(42),
			Validate[int](ozzo.Required, ozzo.Min(0), ozzo.Max(100)),
		),
	)
	is.Nil(err)
	is.Len(intValues, 1)
	is.True(intValues[0].IsOk())
	is.Equal(42, intValues[0].Unwrap())

	// Test with invalid int
	intValues, err = ro.Collect(
		ro.Pipe1(
			ro.Just(150),
			Validate[int](ozzo.Required, ozzo.Min(0), ozzo.Max(100)),
		),
	)
	is.Nil(err)
	is.Len(intValues, 1)
	is.True(intValues[0].IsError())
}

func TestValidate_multipleRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	// Test with multiple validation rules
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("valid@email.com"),
			Validate[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("valid@email.com", values[0].Unwrap())

	// Test with multiple rules where one fails
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("invalid-email"),
			Validate[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())
}

func TestValidate_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test that error observables propagate errors correctly
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			Validate[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidateWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test successful validation with simple string
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("test"),
			ValidateWithContext[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())

	// Test failed validation with simple string
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just(""),
			ValidateWithContext[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsError())
	is.NotNil(values[0].Error())
}

func TestValidateWithContext_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	ctx := context.Background()

	// Test with URL validation
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("https://example.com"),
			ValidateWithContext[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("https://example.com", values[0].Unwrap())

	// Test with invalid URL
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("not-a-url"),
			ValidateWithContext[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())
}

func TestValidateWithContext_complexValidation(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	ctx := context.Background()

	// Test with complex validation rules
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("valid@email.com"),
			ValidateWithContext[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsOk())
	assert.Equal("valid@email.com", values[0].Unwrap())

	// Test with complex validation rules that fail
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("invalid-email"),
			ValidateWithContext[string](
				ozzo.Required,
				is.Email,
				ozzo.Length(1, 50),
			),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.True(values[0].IsError())
}

func TestValidateWithContext_edgeCases(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test with nil rules
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("test"),
			ValidateWithContext[string](),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())

	// Test with multiple rules
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("test"),
			ValidateWithContext[string](ozzo.Required, ozzo.Length(1, 10), ozzo.Match(regexp.MustCompile(`^[a-z]+$`))),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.True(values[0].IsOk())
	is.Equal("test", values[0].Unwrap())
}

func TestValidateWithContext_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test that error observables propagate errors correctly
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			ValidateWithContext[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidateOrError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test successful validation
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("test"),
			ValidateOrError[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.Equal("test", values[0])

	// Test failed validation
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just(""),
			ValidateOrError[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.NotNil(err)
	is.Empty(values)
}

func TestValidateOrError_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	// Test with URL validation
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just("https://example.com"),
			ValidateOrError[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.Equal("https://example.com", values[0])

	// Test with invalid URL
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Just("not-a-url"),
			ValidateOrError[string](ozzo.Required, is.URL),
		),
	)
	assert.NotNil(err)
	assert.Empty(values)
}

func TestValidateOrError_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test that error observables propagate errors correctly
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			ValidateOrError[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidateOrErrorWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test successful validation
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("test"),
			ValidateOrErrorWithContext[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 1)
	is.Equal("test", values[0])

	// Test failed validation
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just(""),
			ValidateOrErrorWithContext[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.NotNil(err)
	is.Empty(values)
}

func TestValidateOrErrorWithContext_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	ctx := context.Background()

	// Test with URL validation
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("https://example.com"),
			ValidateOrErrorWithContext[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 1)
	assert.Equal("https://example.com", values[0])

	// Test with invalid URL
	values, ctx, err = ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just("not-a-url"),
			ValidateOrErrorWithContext[string](ozzo.Required, is.URL),
		),
	)
	assert.NotNil(err)
	assert.Empty(values)
}

func TestValidateOrErrorWithContext_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test that error observables propagate errors correctly
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			ValidateOrErrorWithContext[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidateOrSkip(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test with valid and invalid strings
	strings := []string{"test", "", "valid", "too-long-string-that-exceeds-limit"}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.FromSlice(strings),
			ValidateOrSkip[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 2) // Only valid strings should pass through
	is.Equal("test", values[0])
	is.Equal("valid", values[1])

	// Test empty observable
	values, err = ro.Collect(
		ro.Pipe1(
			ro.Empty[string](),
			ValidateOrSkip[string](ozzo.Required),
		),
	)
	is.Nil(err)
	is.Empty(values)
}

func TestValidateOrSkip_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	// Test with URL validation
	urls := []string{"https://example.com", "not-a-url", "https://google.com", "invalid"}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.FromSlice(urls),
			ValidateOrSkip[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 2) // Only valid URLs should pass through
	assert.Equal("https://example.com", values[0])
	assert.Equal("https://google.com", values[1])
}

func TestValidateOrSkip_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Test that error observables propagate errors correctly
	values, err := ro.Collect(
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			ValidateOrSkip[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}

func TestValidateOrSkipWithContext(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test with valid and invalid strings
	strings := []string{"test", "", "valid", "too-long-string-that-exceeds-limit"}

	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.FromSlice(strings),
			ValidateOrSkipWithContext[string](ozzo.Required, ozzo.Length(1, 10)),
		),
	)
	is.Nil(err)
	is.Len(values, 2) // Only valid strings should pass through
	is.Equal("test", values[0])
	is.Equal("valid", values[1])
}

func TestValidateOrSkipWithContext_additionalValidationRules(t *testing.T) {
	t.Parallel()
	assert := assert.New(t)

	ctx := context.Background()

	// Test with URL validation
	urls := []string{"https://example.com", "not-a-url", "https://google.com", "invalid"}

	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.FromSlice(urls),
			ValidateOrSkipWithContext[string](ozzo.Required, is.URL),
		),
	)
	assert.Nil(err)
	assert.Len(values, 2) // Only valid URLs should pass through
	assert.Equal("https://example.com", values[0])
	assert.Equal("https://google.com", values[1])
}

func TestValidateOrSkipWithContext_errorObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	ctx := context.Background()

	// Test that error observables propagate errors correctly
	values, ctx, err := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			ValidateOrSkipWithContext[string](ozzo.Required),
		),
	)
	is.EqualError(err, assert.AnError.Error())
	is.Empty(values)
}
