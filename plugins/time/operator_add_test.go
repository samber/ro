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

type timeAddTest struct {
	input    time.Time
	duration time.Duration
	expected time.Time
}

var addTests = []timeAddTest{
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		duration: 2 * time.Hour,
		expected: time.Date(2026, time.January, 7, 16, 30, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
		duration: -24 * time.Hour,
		expected: time.Date(2025, time.December, 24, 0, 0, 0, 0, time.UTC),
	},
	{
		input:    time.Date(0, time.January, 1, 23, 59, 59, 0, time.UTC),
		duration: time.Second,
		expected: time.Date(0, time.January, 1, 23, 59, 59+1, 0, time.UTC),
	},
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		duration: 0,
		expected: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
	},
}

type timeAddDateTest struct {
	input    time.Time
	years    int
	months   int
	days     int
	expected time.Time
}

var addDateTests = []timeAddDateTest{
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		years:    0,
		months:   1,
		days:     0,
		expected: time.Date(2026, time.February, 7, 14, 30, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2025, time.December, 25, 0, 0, 0, 0, time.UTC),
		years:    -1,
		months:   0,
		days:     0,
		expected: time.Date(2024, time.December, 25, 0, 0, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		years:    0,
		months:   0,
		days:     5,
		expected: time.Date(2026, time.January, 12, 14, 30, 0, 0, time.UTC),
	},
	{
		input:    time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
		years:    0,
		months:   0,
		days:     0,
		expected: time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC),
	},
}

func TestAdd(t *testing.T) {
	t.Run("Test Simple cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		for _, tt := range addTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					Add(tt.duration),
				),
			)
			is.Nil(err)
			is.Equal([]time.Time{tt.expected}, values)
		}
	})

	t.Run("Test empty observable case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[time.Time](),
				Add(2*time.Hour),
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
				ro.Throw[time.Time](assert.AnError),
				Add(2*time.Hour),
			),
		)
		is.Equal([]time.Time{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}

func TestAddDate(t *testing.T) {
	t.Run("Test Simple cases", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		for _, tt := range addDateTests {
			values, err := ro.Collect(
				ro.Pipe1(
					ro.Just(tt.input),
					AddDate(tt.years, tt.months, tt.days),
				),
			)
			is.Nil(err)
			is.Equal([]time.Time{tt.expected}, values)
		}
	})

	t.Run("Test empty observable case", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Empty[time.Time](),
				AddDate(0, 0, 1),
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
				ro.Throw[time.Time](assert.AnError),
				AddDate(0, 0, 1),
			),
		)
		is.Equal([]time.Time{}, values)
		is.EqualError(err, assert.AnError.Error())
	})
}
