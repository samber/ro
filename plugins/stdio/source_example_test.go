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
	"os"
	"strings"

	"github.com/samber/ro"
)

func ExampleNewIOReader() {
	// Read data from a string reader
	data := "Hello, World! This is a test."
	reader := strings.NewReader(data)
	observable := NewIOReader(reader)

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [72 101 108 108 111 44 32 87 111 114 108 100 33 32 84 104 105 115 32 105 115 32 97 32 116 101 115 116 46]
	// Completed
}

func ExampleNewIOReaderLine() {
	// Read lines from a string reader
	data := "Line 1\nLine 2\nLine 3\n"
	reader := strings.NewReader(data)
	observable := NewIOReaderLine(reader)

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [76 105 110 101 32 49]
	// Next: [76 105 110 101 32 50]
	// Next: [76 105 110 101 32 51]
	// Completed
}

func ExampleNewStdReader() {
	// Read from standard input
	// Simulate stdin input by temporarily redirecting stdin
	originalStdin := os.Stdin
	defer func() { os.Stdin = originalStdin }()

	// Create a pipe to simulate stdin
	r, w, _ := os.Pipe()
	os.Stdin = r

	// Write test data to stdin
	go func() {
		w.WriteString("Hello from stdin!")
		w.Close()
	}()

	observable := NewStdReader()

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [72 101 108 108 111 32 102 114 111 109 32 115 116 100 105 110 33]
	// Completed
}

func ExampleNewStdReaderLine() {
	// Read lines from standard input
	// Simulate stdin input by temporarily redirecting stdin
	originalStdin := os.Stdin
	defer func() { os.Stdin = originalStdin }()

	// Create a pipe to simulate stdin
	r, w, _ := os.Pipe()
	os.Stdin = r

	// Write test data to stdin
	go func() {
		w.WriteString("Line 1\nLine 2\nLine 3\n")
		w.Close()
	}()

	observable := NewStdReaderLine()

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [76 105 110 101 32 49]
	// Next: [76 105 110 101 32 50]
	// Next: [76 105 110 101 32 51]
	// Completed
}

func ExampleNewIOReader_withError() {
	// Read data with potential errors
	reader := strings.NewReader("Hello, World!")
	observable := NewIOReader(reader)

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [72 101 108 108 111 44 32 87 111 114 108 100 33]
	// Completed
}

func ExampleNewIOReaderLine_withLargeFile() {
	// Read lines from a large text
	data := `Line 1: This is the first line
Line 2: This is the second line
Line 3: This is the third line
Line 4: This is the fourth line
Line 5: This is the fifth line`

	reader := strings.NewReader(data)
	observable := NewIOReaderLine(reader)

	subscription := observable.Subscribe(ro.PrintObserver[[]byte]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [76 105 110 101 32 49 58 32 84 104 105 115 32 105 115 32 116 104 101 32 102 105 114 115 116 32 108 105 110 101]
	// Next: [76 105 110 101 32 50 58 32 84 104 105 115 32 105 115 32 116 104 101 32 115 101 99 111 110 100 32 108 105 110 101]
	// Next: [76 105 110 101 32 51 58 32 84 104 105 115 32 105 115 32 116 104 101 32 116 104 105 114 100 32 108 105 110 101]
	// Next: [76 105 110 101 32 52 58 32 84 104 105 115 32 105 115 32 116 104 101 32 102 111 117 114 116 104 32 108 105 110 101]
	// Next: [76 105 110 101 32 53 58 32 84 104 105 115 32 105 115 32 116 104 101 32 102 105 102 116 104 32 108 105 110 101]
	// Completed
}
