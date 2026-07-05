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


package rostrconv

import (
	"github.com/samber/ro"
)

func ExampleAtoi() {
	// Convert strings to integers
	observable := ro.Pipe1(
		ro.Just("123", "456", "789"),
		Atoi[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 123
	// Next: 456
	// Next: 789
	// Completed
}

func ExampleParseInt() {
	// Parse strings as integers with different bases
	observable := ro.Pipe1(
		ro.Just("123", "FF", "1010"),
		ParseInt[string](16, 64), // Parse as hex, 64-bit
	)

	subscription := observable.Subscribe(ro.PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 291
	// Next: 255
	// Next: 4112
	// Completed
}

func ExampleParseFloat() {
	// Parse strings as floats
	observable := ro.Pipe1(
		ro.Just("3.14", "2.718", "1.414"),
		ParseFloat[string](64), // Parse as 64-bit float
	)

	subscription := observable.Subscribe(ro.PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3.14
	// Next: 2.718
	// Next: 1.414
	// Completed
}

func ExampleParseBool() {
	// Parse strings as booleans
	observable := ro.Pipe1(
		ro.Just("true", "false", "1", "0"),
		ParseBool[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[bool]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: true
	// Next: false
	// Next: true
	// Next: false
	// Completed
}

func ExampleParseUint() {
	// Parse strings as unsigned integers
	observable := ro.Pipe1(
		ro.Just("123", "456", "789"),
		ParseUint[string](10, 64), // Parse as decimal, 64-bit unsigned
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 123
	// Next: 456
	// Next: 789
	// Completed
}

func ExampleFormatBool() {
	// Convert booleans to strings
	observable := ro.Pipe1(
		ro.Just(true, false, true),
		FormatBool(),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: true
	// Next: false
	// Next: true
	// Completed
}

func ExampleFormatFloat() {
	// Convert floats to strings
	observable := ro.Pipe1(
		ro.Just(3.14159, 2.71828, 1.41421),
		FormatFloat('f', 3, 64), // Format with 3 decimal places
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3.142
	// Next: 2.718
	// Next: 1.414
	// Completed
}

func ExampleFormatInt() {
	// Convert integers to strings with different bases
	observable := ro.Pipe1(
		ro.Just(int64(255), int64(123), int64(456)),
		FormatInt[string](16), // Format as hex
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: ff
	// Next: 7b
	// Next: 1c8
	// Completed
}

func ExampleFormatUint() {
	// Convert unsigned integers to strings
	observable := ro.Pipe1(
		ro.Just(uint64(255), uint64(123), uint64(456)),
		FormatUint[string](10), // Format as decimal
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 255
	// Next: 123
	// Next: 456
	// Completed
}

func ExampleItoa() {
	// Convert integers to strings
	observable := ro.Pipe1(
		ro.Just(123, 456, 789),
		Itoa(),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 123
	// Next: 456
	// Next: 789
	// Completed
}

func ExampleQuote() {
	// Quote strings
	observable := ro.Pipe1(
		ro.Just("hello", "world", "golang"),
		Quote(),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: "hello"
	// Next: "world"
	// Next: "golang"
	// Completed
}

func ExampleQuoteRune() {
	// Quote runes
	observable := ro.Pipe1(
		ro.Just('a', 'b', 'c'),
		QuoteRune(),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 'a'
	// Next: 'b'
	// Next: 'c'
	// Completed
}

func ExampleUnquote() {
	// Unquote strings
	observable := ro.Pipe1(
		ro.Just(`"hello"`, `"world"`, `"golang"`),
		Unquote(),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: hello
	// Next: world
	// Next: golang
	// Completed
}

func ExampleParseInt_withError() {
	// Parse strings with potential errors
	observable := ro.Pipe1(
		ro.Just("123", "abc", "456"), // "abc" will cause an error
		ParseInt[string](10, 64),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 123
	// Error: strconv.ParseInt: parsing "abc": invalid syntax
}

func ExampleParseUint64() {
	// Parse strings as unsigned 64-bit integers
	observable := ro.Pipe1(
		ro.Just("42", "100", "255"),
		ParseUint64[string](10, 64),
	)

	subscription := observable.Subscribe(ro.PrintObserver[uint64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 42
	// Next: 100
	// Next: 255
	// Completed
}

func ExampleFormatComplex() {
	// Convert complex128 values to strings with fixed-point notation
	observable := ro.Pipe1(
		ro.Just(complex(3, 4), complex(1, 2), complex(0, -1)),
		FormatComplex('f', 2, 128),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: (3.00+4.00i)
	// Next: (1.00+2.00i)
	// Next: (0.00-1.00i)
	// Completed
}
