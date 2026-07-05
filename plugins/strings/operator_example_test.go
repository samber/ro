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

package rostrings

import (
	"github.com/samber/ro"
	"golang.org/x/text/language"
)

func ExampleCamelCase() {
	// Convert strings to camelCase format
	observable := ro.Pipe1(
		ro.Just(
			"hello world",
			"user_name",
			"API_KEY",
			"camel case",
		),
		CamelCase[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: helloWorld
	// Next: userName
	// Next: apiKey
	// Next: camelCase
	// Completed
}

func ExampleCapitalize() {
	// Capitalize the first letter of each string
	observable := ro.Pipe1(
		ro.Just(
			"hello",
			"world",
			"golang",
		),
		Capitalize[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hello
	// Next: World
	// Next: Golang
	// Completed
}

func ExampleCapitalizeWithLanguage() {
	obs := ro.Pipe[string, string](
		ro.Just("istanbul"),
		CapitalizeWithLanguage[string](language.Turkish),
	)

	sub := obs.Subscribe(ro.PrintObserver[string]())
	defer sub.Unsubscribe()

	// Output:
	// Next: İstanbul
	// Completed
}

func ExampleKebabCase() {
	// Convert strings to kebab-case format
	observable := ro.Pipe1(
		ro.Just(
			"hello world",
			"userName",
			"API_KEY",
			"camelCase",
		),
		KebabCase[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: hello-world
	// Next: user-name
	// Next: api-key
	// Next: camel-case
	// Completed
}

func ExamplePascalCase() {
	// Convert strings to PascalCase format
	observable := ro.Pipe1(
		ro.Just(
			"hello world",
			"user_name",
			"api_key",
			"camel case",
		),
		PascalCase[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: HelloWorld
	// Next: UserName
	// Next: ApiKey
	// Next: CamelCase
	// Completed
}

func ExampleSnakeCase() {
	// Convert strings to snake_case format
	observable := ro.Pipe1(
		ro.Just(
			"hello world",
			"userName",
			"API_KEY",
			"camelCase",
		),
		SnakeCase[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: hello_world
	// Next: user_name
	// Next: api_key
	// Next: camel_case
	// Completed
}

func ExampleEllipsis() {
	// Truncate strings with ellipsis
	observable := ro.Pipe1(
		ro.Just(
			"This is a very long string that needs to be truncated",
			"Short",
			"Another long string for demonstration",
		),
		Ellipsis[string](20),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: This is a very lo...
	// Next: Short
	// Next: Another long stri...
	// Completed
}

func ExampleWords() {
	// Split strings into words
	observable := ro.Pipe1(
		ro.Just(
			"hello world",
			"user_name",
			"camelCase",
			"PascalCase",
		),
		Words[string](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[[]string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [hello world]
	// Next: [user name]
	// Next: [camel Case]
	// Next: [Pascal Case]
	// Completed
}

func ExampleRandom() {
	// Generate random strings
	observable := ro.Pipe1(
		ro.Just(1, 2, 3),
		Random[int](10, AlphanumericCharset),
	)

	subscription := observable.Subscribe(ro.NoopObserver[string]())
	defer subscription.Unsubscribe()
}

func ExampleCamelCaseWithLanguage() {
	// Convert strings to camelCase using locale-aware casing (Turkish)
	obs := ro.Pipe1(
		ro.Just("Istanbul city"),
		CamelCaseWithLanguage[string](language.Turkish),
	)

	sub := obs.Subscribe(ro.PrintObserver[string]())
	defer sub.Unsubscribe()

	// Output:
	// Next: ıstanbulCity
	// Completed
}

func ExampleKebabCaseWithLanguage() {
	// Convert strings to kebab-case using locale-aware lowercasing (Turkish)
	obs := ro.Pipe1(
		ro.Just("IstanbulCity"),
		KebabCaseWithLanguage[string](language.Turkish),
	)

	sub := obs.Subscribe(ro.PrintObserver[string]())
	defer sub.Unsubscribe()

	// Output:
	// Next: ıstanbul-city
	// Completed
}

func ExamplePascalCaseWithLanguage() {
	// Convert strings to PascalCase using locale-aware casing (Turkish)
	obs := ro.Pipe1(
		ro.Just("istanbul city"),
		PascalCaseWithLanguage[string](language.Turkish),
	)

	sub := obs.Subscribe(ro.PrintObserver[string]())
	defer sub.Unsubscribe()

	// Output:
	// Next: İstanbulCity
	// Completed
}

func ExampleSnakeCaseWithLanguage() {
	// Convert strings to snake_case using locale-aware lowercasing (Turkish)
	obs := ro.Pipe1(
		ro.Just("IstanbulCity"),
		SnakeCaseWithLanguage[string](language.Turkish),
	)

	sub := obs.Subscribe(ro.PrintObserver[string]())
	defer sub.Unsubscribe()

	// Output:
	// Next: ıstanbul_city
	// Completed
}
