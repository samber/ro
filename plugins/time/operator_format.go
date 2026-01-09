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

// Format returns a function that transforms an observable of time.Time values
// into an observable of strings, formatted according to the provided layout.
// The layout must be a valid time format string (e.g., "2006-01-02 15:04:05").
func Format(format string) func(destination ro.Observable[time.Time]) ro.Observable[string] {
	return ro.Map(
		func(value time.Time) string {
			return value.Format(format)
		},
	)
}

//obs := ro.Pipe(
//    ro.Just(time.Now()),
//    rotime.Format("2006-01-02 15:04:05"),
//)
// obs now emits the current time as a formatted string, e.g., "2026-01-07 14:30:00"
