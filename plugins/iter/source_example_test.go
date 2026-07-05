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


package roiter

import (
	"github.com/samber/lo"
	"github.com/samber/ro"
)

func ExampleFromSeq() {
	// Create a sequence of integers
	seq := func(yield func(int) bool) {
		for i := 1; i <= 5; i++ {
			if !yield(i) {
				return
			}
		}
	}

	observable := FromSeq[int](seq)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleFromSeq2() {
	// Create a sequence of key-value pairs
	seq := func(yield func(string, int) bool) {
		pairs := []struct {
			key   string
			value int
		}{
			{"Alice", 30},
			{"Bob", 25},
			{"Charlie", 35},
		}

		for _, pair := range pairs {
			if !yield(pair.key, pair.value) {
				return
			}
		}
	}

	observable := FromSeq2[string, int](seq)

	subscription := observable.Subscribe(ro.PrintObserver[lo.Tuple2[string, int]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {Alice 30}
	// Next: {Bob 25}
	// Next: {Charlie 35}
	// Completed
}

func ExampleFromSeq_withStrings() {
	// Create a sequence of strings
	seq := func(yield func(string) bool) {
		words := []string{"Hello", "World", "Golang", "Reactive"}
		for _, word := range words {
			if !yield(word) {
				return
			}
		}
	}

	observable := FromSeq[string](seq)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hello
	// Next: World
	// Next: Golang
	// Next: Reactive
	// Completed
}

