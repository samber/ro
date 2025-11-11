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

package rostdio

import (
	"bytes"

	"github.com/samber/ro"
)

func ExampleNewIOWriter() {
	// Write data to a buffer
	var buf bytes.Buffer
	writer := &buf

	data := ro.Just(
		[]byte("Hello, "),
		[]byte("World!"),
	)

	observable := ro.Pipe1(
		data,
		NewIOWriter(writer),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 13
	// Completed
}

func ExampleNewStdWriter() {
	// Write data to standard output
	// For this example, we'll use a buffer to simulate stdout
	var buf bytes.Buffer

	data := ro.Just(
		[]byte("Hello, "),
		[]byte("World!"),
	)

	observable := ro.Pipe1(
		data,
		NewIOWriter(&buf),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 13
	// Completed
}

func ExampleNewIOWriter_withError() {
	// Write data with potential errors
	var buf bytes.Buffer
	writer := &buf

	data := ro.Just(
		[]byte("Hello, "),
		[]byte("World!"),
	)

	observable := ro.Pipe1(
		data,
		NewIOWriter(writer),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 13
	// Completed
}
