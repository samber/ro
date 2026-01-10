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
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type timeFormatTest struct {
	input    time.Time
	format   string
	expected string
}

var allTimeFormatTests = []timeFormatTest{
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		format:   "2006-01-02 15:04:05",
		expected: "2026-01-07 14:30:00",
	},
	{
		input:    time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
		format:   "2006/01/02",
		expected: "2025/12/25",
	},
	{
		input:    time.Date(2024, time.November, 1, 23, 59, 59, 0, time.UTC),
		format:   "15:04:05",
		expected: "23:59:59",
	},
}

func TestFormat(t *testing.T) {
	t.Run("Simple test cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		for _, tt := range allTimeFormatTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					Format(tt.format),
				),
			)
			is.Equal([]string{tt.expected}, values)
			is.Nil(err)
		}
	})

	t.Run("Test empty observable", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[time.Time](),
				Format("2006-01-02 15:04:05"),
			),
		)
		is.Equal([]string{}, values)
		is.Nil(err)
	})

	t.Run("Test error handling", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Throw[time.Time](assert.AnError),
				Format("2006-01-02 15:04:05"),
			),
		)
		is.Equal([]string{}, values)
		is.EqualError(err, assert.AnError.Error())
	})

}
