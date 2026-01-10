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

type timeParseTest struct {
	input    string
	format   string
	expected time.Time
	loc      *time.Location
}

var parseTests = []timeParseTest{
	{
		expected: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		format:   "2006-01-02 15:04:05",
		input:    "2026-01-07 14:30:00",
	},
	{
		expected: time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
		format:   "2006/01/02",
		input:    "2025/12/25",
	},
	{
		expected: time.Date(0, time.January, 1, 23, 59, 59, 0, time.UTC),
		format:   "15:04:05",
		input:    "23:59:59",
	},
}

var parseInLocationTests = []timeParseTest{
	{
		expected: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		format:   "2006-01-02 15:04:05",
		input:    "2026-01-07 14:30:00",
		loc:      time.UTC,
	},
	{
		expected: time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
		format:   "2006/01/02",
		input:    "2025/12/25",
		loc:      time.UTC,
	},
	{
		expected: time.Date(0, time.January, 1, 23, 59, 59, 0, time.UTC),
		format:   "15:04:05",
		input:    "23:59:59",
		loc:      time.UTC,
	},
	{
		input:  "2026-01-07 14:30:00",
		format: "2006-01-02 15:04:05",
		loc:    time.FixedZone("GMT+2", 2*60*60),
		expected: time.Date(
			2026, time.January, 7, 14, 30, 0, 0, time.FixedZone("GMT+2", 2*60*60),
		),
	},
}

func TestParse(t *testing.T) {
	t.Run("Test Simple cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		for _, tt := range parseTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					Parse[string](tt.format),
				),
			)
			is.Equal([]time.Time{tt.expected}, values)
			if tt.input == "not-a-date" {
				is.Error(err)
			} else {
				is.NoError(err)
			}
		}
	})

	t.Run("Test empty obsersable case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[string](),
				Parse[string]("2006-01-02 15:04:05"),
			),
		)
		is.Equal([]time.Time{}, values)
		is.NoError(err)
	})

	t.Run("Test error handling case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Throw[string](assert.AnError),
				Parse[string]("2006-01-02 15:04:05"),
			),
		)
		is.Equal([]time.Time{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestParseInLocation(t *testing.T) {
	t.Run("Test Simple cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		for _, tt := range parseInLocationTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					ParseInLocation[string](tt.format, tt.loc),
				),
			)

			is.Nil(err)
			is.Equal([]time.Time{tt.expected}, values)
		}
	})
	t.Run("Test empty obsersable case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[string](),
				ParseInLocation[string]("2006-01-02 15:04:05", time.UTC),
			),
		)

		is.Nil(err)
		is.Equal([]time.Time{}, values)
	})

	t.Run("Test error handling case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Throw[string](assert.AnError),
				ParseInLocation[string]("2006-01-02 15:04:05", time.UTC),
			),
		)

		is.Equal([]time.Time{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}
