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


package rosort

import (
	"strings"
	"time"

	"github.com/samber/ro"
)

func ExampleSort() {
	// Sort values using the default comparison function for ordered types
	observable := ro.Pipe1(
		ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
		Sort(func(a, b int) int {
			return a - b
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 9
	// Completed
}

func ExampleSortFunc() {
	// Sort values using a custom comparison function
	type User struct {
		Name string
		Age  int
	}

	observable := ro.Pipe1(
		ro.Just(
			User{Name: "Alice", Age: 30},
			User{Name: "Bob", Age: 25},
			User{Name: "Charlie", Age: 35},
		),
		SortFunc(func(a, b User) int {
			return a.Age - b.Age // Sort by age
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[User]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {Bob 25}
	// Next: {Alice 30}
	// Next: {Charlie 35}
	// Completed
}

func ExampleSortStableFunc() {
	// Sort values using a custom comparison function with stable sorting
	type Event struct {
		Timestamp time.Time
		Priority  int
		Message   string
	}

	observable := ro.Pipe2(
		ro.Just(
			Event{Timestamp: time.Now(), Priority: 1, Message: "First"},
			Event{Timestamp: time.Now().Add(time.Second), Priority: 2, Message: "Second"},
			Event{Timestamp: time.Now().Add(2 * time.Second), Priority: 1, Message: "Third"},
		),
		SortStableFunc(func(a, b Event) int {
			return a.Priority - b.Priority // Sort by priority, stable
		}),
		ro.Map(func(event Event) Event {
			return Event{Priority: event.Priority, Message: event.Message}
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Event]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {0001-01-01 00:00:00 +0000 UTC 1 First}
	// Next: {0001-01-01 00:00:00 +0000 UTC 1 Third}
	// Next: {0001-01-01 00:00:00 +0000 UTC 2 Second}
	// Completed
}

func ExampleSortFunc_complexLogic() {
	// Sort with complex logic
	type Product struct {
		Name     string
		Price    float64
		Category string
	}

	observable := ro.Pipe1(
		ro.Just(
			Product{Name: "Laptop", Price: 999.99, Category: "Electronics"},
			Product{Name: "Book", Price: 19.99, Category: "Books"},
			Product{Name: "Phone", Price: 699.99, Category: "Electronics"},
			Product{Name: "Pen", Price: 2.99, Category: "Office"},
		),
		SortFunc(func(a, b Product) int {
			// Sort by category first, then by price
			if a.Category != b.Category {
				return strings.Compare(a.Category, b.Category)
			}
			if a.Price < b.Price {
				return -1
			}
			if a.Price > b.Price {
				return 1
			}
			return 0
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[Product]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {Book 19.99 Books}
	// Next: {Phone 699.99 Electronics}
	// Next: {Laptop 999.99 Electronics}
	// Next: {Pen 2.99 Office}
	// Completed
}

func ExampleSort_sortingStrings() {
	// Sort strings
	observable := ro.Pipe1(
		ro.Just("zebra", "apple", "banana", "cherry"),
		Sort(func(a, b string) int {
			return strings.Compare(a, b)
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: apple
	// Next: banana
	// Next: cherry
	// Next: zebra
	// Completed
}

func ExampleReverse() {
	// Buffer all items until completion, then re-emit them in reverse order
	observable := ro.Pipe1(
		ro.Just(1, 2, 3, 4, 5),
		Reverse[int](),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Next: 4
	// Next: 3
	// Next: 2
	// Next: 1
	// Completed
}

func ExampleSortFunc_sortingWithCustomStringLogic() {
	// Sort strings by length, then alphabetically
	observable := ro.Pipe1(
		ro.Just("cat", "dog", "elephant", "ant", "bird"),
		SortFunc(func(a, b string) int {
			// Sort by length first
			if len(a) != len(b) {
				return len(a) - len(b)
			}
			// Then alphabetically
			return strings.Compare(a, b)
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: ant
	// Next: cat
	// Next: dog
	// Next: bird
	// Next: elephant
	// Completed
}
