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

// Parse returns a function that transforms an observable of string-like values
// into an observable of time.Time. On parse error, the zero time is emitted.
func Parse[T ~string](layout string) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.Map(
		func(value T) time.Time {
			t, err := time.Parse(layout, string(value))
			if err != nil {
				return time.Time{}
			}
			return t
		},
	)
}

func ParseInLocation[T ~string](layout string, loc *time.Location) func(ro.Observable[T]) ro.Observable[time.Time] {
	return ro.Map(
		func(value T) time.Time {
			t, err := time.ParseInLocation(layout, string(value), loc)
			if err != nil {
				return time.Time{}
			}
			return t
		},
	)
}
