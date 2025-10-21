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

package roics

import (
	ics "github.com/arran4/golang-ical"
	"github.com/samber/ro"
)

func ExampleNewICSFileReader() {
	obs := ro.Pipe1(
		NewICSFileReader(
			"testdata/fr-public-holidays-a.ics",
			"testdata/fr-public-holidays-b.ics",
			"testdata/fr-public-holidays-c.ics",
		),
		ro.Count[*ics.VEvent](),
	)

	subscription := obs.Subscribe(ro.PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 183
	// Completed
}

func ExampleNewICSURLReader() {
	obs := ro.Pipe1(
		NewICSURLReader(
			"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-a.ics",
			"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-b.ics",
			"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-c.ics",
		),
		ro.Count[*ics.VEvent](),
	)
	subscription := obs.Subscribe(ro.PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 183
	// Completed
}
