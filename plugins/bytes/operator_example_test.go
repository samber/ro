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

package robytes

import (
	"fmt"
	"strings"

	"github.com/samber/ro"
	"golang.org/x/text/language"
)

func ExampleCamelCase() {
	// Convert strings to camelCase format
	observable := ro.Pipe1(
		ro.Just(
			[]byte("hello world"),
			[]byte("user_name"),
			[]byte("API_KEY"),
			[]byte("camel case"),
		),
		CamelCase[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
			[]byte("hello"),
			[]byte("world"),
			[]byte("golang"),
		),
		Capitalize[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hello
	// Next: World
	// Next: Golang
	// Completed
}

func ExampleCapitalizeWithLanguage() {
	observable := ro.Pipe1(
		ro.Just([]byte("istanbul")),
		CapitalizeWithLanguage[[]byte](language.Turkish),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Next: İstanbul
	// Completed
}

func ExampleKebabCase() {
	// Convert strings to kebab-case format
	observable := ro.Pipe1(
		ro.Just(
			[]byte("hello world"),
			[]byte("userName"),
			[]byte("API_KEY"),
			[]byte("camelCase"),
		),
		KebabCase[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
			[]byte("hello world"),
			[]byte("user_name"),
			[]byte("api_key"),
			[]byte("camel case"),
		),
		PascalCase[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
			[]byte("hello world"),
			[]byte("userName"),
			[]byte("API_KEY"),
			[]byte("camelCase"),
		),
		SnakeCase[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
			[]byte("This is a very long string that needs to be truncated"),
			[]byte("Short"),
			[]byte("Another long string for demonstration"),
		),
		Ellipsis[[]byte](20),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data []byte) {
				fmt.Printf("Next: %s\n", string(data))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
			[]byte("hello world"),
			[]byte("user_name"),
			[]byte("camelCase"),
			[]byte("PascalCase"),
		),
		Words[[]byte](),
	)

	subscription := observable.Subscribe(
		ro.NewObserver(
			func(data [][]byte) {
				words := make([]string, len(data))
				for i, word := range data {
					words[i] = string(word)
				}
				fmt.Printf("Next: [%s]\n", strings.Join(words, " "))
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)
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
		ro.Just(
			[]byte("prefix"),
			[]byte("suffix"),
			[]byte("base"),
		),
		Random[[]byte](10, AlphanumericCharset),
	)

	subscription := observable.Subscribe(ro.NoopObserver[[]byte]())
	defer subscription.Unsubscribe()
}
