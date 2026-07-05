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

package rotime

import (
	"time"

	"github.com/samber/ro"
)

func ExampleAdd() {
	t := time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
	observable := ro.Pipe2(
		ro.Just(t),
		Add(2*time.Hour),
		Format("2006-01-02 15:04:05"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 16:30:00
	// Completed
}

func ExampleAddDate() {
	t := time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
	observable := ro.Pipe2(
		ro.Just(t),
		AddDate(0, 1, 0),
		Format("2006-01-02 15:04:05"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-02-07 14:30:00
	// Completed
}

func ExampleFormat() {
	t := time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
	observable := ro.Pipe1(
		ro.Just(t),
		Format("2006-01-02 15:04:05"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 14:30:00
	// Completed
}

func ExampleIn() {
	loc := time.FixedZone("UTC+2", 2*60*60)
	t := time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
	observable := ro.Pipe2(
		ro.Just(t),
		In(loc),
		Format("2006-01-02 15:04:05 -0700"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 16:30:00 +0200
	// Completed
}

func ExampleParse() {
	observable := ro.Pipe2(
		ro.Just("2026-01-07 14:30:00"),
		Parse[string]("2006-01-02 15:04:05"),
		Format("2006-01-02 15:04:05"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 14:30:00
	// Completed
}

func ExampleParseInLocation() {
	loc := time.FixedZone("UTC+5", 5*60*60)
	observable := ro.Pipe2(
		ro.Just("2026-01-07 14:30:00"),
		ParseInLocation[string]("2006-01-02 15:04:05", loc),
		Format("2006-01-02 15:04:05 -0700"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 14:30:00 +0500
	// Completed
}

func ExampleStartOfDay() {
	t := time.Date(2026, time.January, 7, 14, 30, 45, 0, time.UTC)
	observable := ro.Pipe2(
		ro.Just(t),
		StartOfDay(),
		Format("2006-01-02 15:04:05"),
	)

	subscription := observable.Subscribe(ro.PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2026-01-07 00:00:00
	// Completed
}
