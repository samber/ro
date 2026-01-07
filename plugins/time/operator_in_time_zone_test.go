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

func TestInTimeZone_SimpleConversion(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	utc := time.Date(2026, time.January, 7, 14, 30, 0, 0, time.UTC)
	paris, _ := time.LoadLocation("Europe/Paris")

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just(utc),
			InTimeZone(paris),
		),
	)
	is.NoError(err)

	got := values[0]

	// Same instant.
	is.True(utc.Equal(got))

	// Different location (zone name / offset).
	name, _ := got.Zone()
	is.Equal("CET", name)
}

func TestInTimeZone_EmptyObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Empty[time.Time](),
			InTimeZone(time.UTC),
		),
	)

	is.NoError(err)
	is.Equal([]time.Time{}, values)
}

func TestInTimeZone_PropagatesError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Throw[time.Time](assert.AnError),
			InTimeZone(time.UTC),
		),
	)

	is.Equal([]time.Time{}, values)
	is.EqualError(err, assert.AnError.Error())
}
