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

	"github.com/samber/ro"
)

type examplePerson struct {
	Name  string `faker:"name"`
	Email string `faker:"email"`
}

func ExampleFake() {
	obs := Fake[examplePerson](2)

	values, _ := ro.Collect(obs)

	fmt.Printf("count: %d\n", len(values))
	fmt.Printf("has name: %v\n", values[0].Name != "")
	fmt.Printf("has email: %v\n", values[0].Email != "")

	// Output:
	// count: 2
	// has name: true
	// has email: true
}

func ExampleEmail() {
	obs := Email(2)

	values, _ := ro.Collect(obs)

	fmt.Printf("count: %d\n", len(values))
	fmt.Printf("has @: %v\n", len(values[0]) > 0)

	// Output:
	// count: 2
	// has @: true
}

func ExampleUUIDHyphenated() {
	obs := UUIDHyphenated(1)

	values, _ := ro.Collect(obs)

	fmt.Printf("count: %d\n", len(values))
	fmt.Printf("has -: %v\n", len(values[0]) > 0)

	// Output:
	// count: 1
	// has -: true
}
