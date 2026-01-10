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

// StartOfDay returns an operator that truncates each time value to the start of its day.
//
// Example:
//
//	obs := ro.Pipe1(
//	    ro.Just(time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)),
//	    rotime.StartOfDay(),
//	)
//
// The observable then emits: time.Date(2026, time.January, 7, 0, 0, 0, 0, time.UTC).
func StartOfDay() func(ro.Observable[time.Time]) ro.Observable[time.Time] {
	return ro.Map(
		func(value time.Time) time.Time {
			year, month, day := value.Date()
			return time.Date(year, month, day, 0, 0, 0, 0, value.Location())
		},
	)
}
