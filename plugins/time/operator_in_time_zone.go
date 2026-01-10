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

// In returns an operator that converts each time value to the given location.
//
// Example:
//
//	loc, _ := time.LoadLocation("Europe/Paris")
//
//	obs := ro.Pipe1(
//	    ro.Just(time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)),
//	    rotime.In(loc),
//	)
//
// The observable then emits: time.Date(2026, time.January, 7, 15, 30, 0, 0, loc).
func In(loc *time.Location) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			return value.In(loc)
		},
	)
}
