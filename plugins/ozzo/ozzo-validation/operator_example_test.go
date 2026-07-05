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
	"fmt"

	ozzo "github.com/go-ozzo/ozzo-validation/v4"
	"github.com/go-ozzo/ozzo-validation/v4/is"
	"github.com/samber/ro"
)

func ExampleValidate() {
	// Validate values using ozzo validation rules
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
			"invalid-email",
		),
		Validate[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Result[string]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {false valid@email.com <nil>}
	// Next: {true  {validation_is_email must be a valid email address map[]}}
	// Completed
}

func ExampleValidateStruct() {
	// Validate structs that implement the Validatable interface
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
			ValidatableUser{Name: "", Email: "invalid-email", Age: 15},
		),
		ValidateStruct[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Result[ValidatableUser]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {false {John john@example.com 25} <nil>}
	// Next: {true {  0} map[age:{validation_min_greater_equal_than_required must be no less than {{.threshold}} map[threshold:18]} email:{validation_is_email must be a valid email address map[]} name:{validation_required cannot be blank map[]}]}
	// Completed
}

func ExampleValidateOrError() {
	// Validate values and propagate errors through the stream
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
			"invalid-email",
		),
		ValidateOrError[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: valid@email.com
	// Error: must be a valid email address
}

func ExampleValidateOrSkip() {
	// Validate values and skip invalid ones
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
			"invalid-email",
			"another@valid.com",
		),
		ValidateOrSkip[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: valid@email.com
	// Next: another@valid.com
	// Completed
}

func ExampleValidateWithContext() {
	// Validate values with context-aware validation
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
		),
		ValidateWithContext[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Result[string]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {false valid@email.com <nil>}
	// Completed
}

func ExampleValidateStructWithContext() {
	// Validate structs with context-aware validation
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
		),
		ValidateStructWithContext[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Result[ValidatableUser]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {false {John john@example.com 25} <nil>}
	// Completed
}

func ExampleValidateOrErrorWithContext() {
	// Validate values with context and propagate errors
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
		),
		ValidateOrErrorWithContext[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: valid@email.com
	// Completed
}

func ExampleValidateOrSkipWithContext() {
	// Validate values with context and skip invalid ones
	observable := ro.Pipe1(
		ro.Just(
			"valid@email.com",
			"invalid-email",
		),
		ValidateOrSkipWithContext[string](
			ozzo.Required,
			is.Email,
			ozzo.Length(1, 50),
		),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: valid@email.com
	// Completed
}

func ExampleOk() {
	// Create a successful Result containing a value
	result := Ok("hello")
	fmt.Println(result.IsOk())
	fmt.Println(result.Unwrap())

	// Output:
	// true
	// hello
}

func ExampleErr() {
	// Create an error Result containing an error
	result := Err[string](errors.New("something went wrong"))
	fmt.Println(result.IsError())
	fmt.Println(result.Error())

	// Output:
	// true
	// something went wrong
}

func ExampleValidateStructOrError() {
	// Validate struct items that implement ozzo.Validatable; valid items pass through, invalid items terminate stream with error
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
			ValidatableUser{Name: "", Email: "invalid-email", Age: 15},
		),
		ValidateStructOrError[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[ValidatableUser]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {John john@example.com 25}
	// Error: age: must be no less than 18; email: must be a valid email address; name: cannot be blank.
}

func ExampleValidateStructOrErrorWithContext() {
	// Validate struct items that implement ozzo.ValidatableWithContext with context propagation; valid items pass through, invalid items terminate stream with error
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
		),
		ValidateStructOrErrorWithContext[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[ValidatableUser]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {John john@example.com 25}
	// Completed
}

func ExampleValidateStructOrSkip() {
	// Validate struct items that implement ozzo.Validatable; valid items pass through, invalid items are silently skipped
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
			ValidatableUser{Name: "", Email: "invalid-email", Age: 15},
			ValidatableUser{Name: "Jane", Email: "jane@example.com", Age: 30},
		),
		ValidateStructOrSkip[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[ValidatableUser]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {John john@example.com 25}
	// Next: {Jane jane@example.com 30}
	// Completed
}

func ExampleValidateStructOrSkipWithContext() {
	// Validate struct items that implement ozzo.ValidatableWithContext with context propagation; valid items pass through, invalid items are silently skipped
	observable := ro.Pipe1(
		ro.Just(
			ValidatableUser{Name: "John", Email: "john@example.com", Age: 25},
			ValidatableUser{Name: "", Email: "invalid-email", Age: 15},
		),
		ValidateStructOrSkipWithContext[ValidatableUser](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[ValidatableUser]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {John john@example.com 25}
	// Completed
}
