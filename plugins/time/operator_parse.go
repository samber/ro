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

// Parse returns an operator that parses time strings using the given layout.
//
// Example:
//
//	obs := ro.Pipe[string, time.Time](
//	    ro.Just("2026-01-07 14:30:00"),
//	    rotime.Parse("2006-01-02 15:04:05"),
//	)
//
// The observable then emits: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC).
func Parse[T ~string](layout string) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.MapErr(
		func(value T) (time.Time, error) {
			return time.Parse(layout, string(value))
		},
	)
}

// ParseInLocation returns an operator that parses time strings in the given location.
//
// Example:
//
//	obs := ro.Pipe[string, time.Time](
//	    ro.Just("2026-01-07 14:30:00"),
//	    rotime.ParseInLocation("2006-01-02 15:04:05", time.UTC),
//	)
//
// The observable then emits: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC).
func ParseInLocation[T ~string](layout string, loc *time.Location) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.MapErr(
		func(value T) (time.Time, error) {
			return time.ParseInLocation(layout, string(value), loc)
		},
	)
}
