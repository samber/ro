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

// Add returns an operator that adds a fixed duration to each time value.
//
// Example:
//
//	obs := ro.Pipe1(
//	    ro.Just(time.Now()),
//	    rotime.Add(2*time.Hour),
//	)
//
// The observable then emits: time.Now().Add(2 * time.Hour).
func Add(d time.Duration) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.Add(d)
		},
	)
}

// AddDate returns an operator that adds a date offset (years, months, days) to each time value.
//
// Example:
//
//	obs := ro.Pipe1(
//	    ro.Just(time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)),
//	    rotime.AddDate(0, 1, 0),
//	)
//
// The observable then emits: time.Date(2026, time.February, 7, 14, 30, 0, 0, time.UTC).
func AddDate(years int, months int, days int) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.AddDate(years, months, days)
		},
	)
}
