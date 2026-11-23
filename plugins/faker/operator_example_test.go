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

package rofaker

import (
	"fmt"
	"strings"

	"github.com/samber/ro"
)

func ExampleFake() {
	type User struct {
		Name  string `faker:"name"`
		Email string `faker:"email"`
	}

	values, err := ro.Collect(Fake[User](3))
	if err != nil {
		panic(err)
	}

	for _, user := range values {
		fmt.Println(user.Name != "", user.Email != "")
	}

	// Output:
	// true true
	// true true
	// true true
}

func ExampleEmail() {
	observable := ro.Pipe1(
		Email(3),
		ro.Map(func(email string) bool {
			return strings.Contains(email, "@")
		}),
	)

	subscription := observable.Subscribe(ro.PrintObserver[bool]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: true
	// Next: true
	// Next: true
	// Completed
}

func ExampleGenerate() {
	counter := 0

	observable := ro.Pipe1(
		Generate(func() (int, error) {
			counter++
			return counter * counter, nil
		}),
		ro.Take[int](4),
	)

	subscription := observable.Subscribe(ro.PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 4
	// Next: 9
	// Next: 16
	// Completed
}
