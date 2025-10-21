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

package roics

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestNewICSFileReader(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// 3 files
	obs := NewICSFileReader(
		"testdata/fr-public-holidays-a.ics",
		"testdata/fr-public-holidays-b.ics",
		"testdata/fr-public-holidays-c.ics",
	)
	items, err := ro.Collect(obs)
	is.NoError(err)
	is.Len(items, 183)

	// empty
	obs = NewICSFileReader()
	items, err = ro.Collect(obs)
	is.NoError(err)
	is.Len(items, 0)

	// file not found
	obs = NewICSFileReader("testdata/not-found.ics")
	items, err = ro.Collect(obs)
	is.ErrorContains(err, "open testdata/not-found.ics: no such file or directory")
	is.Len(items, 0)
}

func TestNewICSURLReader(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// 3 files
	obs := NewICSURLReader(
		"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-a.ics",
		"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-b.ics",
		"https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/fr-public-holidays-c.ics",
	)
	items, err := ro.Collect(obs)
	is.NoError(err)
	is.Len(items, 183)

	// empty
	obs = NewICSURLReader()
	items, err = ro.Collect(obs)
	is.NoError(err)
	is.Len(items, 0)

	// file not found
	obs = NewICSURLReader("https://raw.githubusercontent.com/samber/ro/refs/heads/main/plugins/ics/testdata/not-found.ics")
	items, err = ro.Collect(obs)
	is.ErrorContains(err, "malformed calendar; expected begin")
	is.Len(items, 0)
}
