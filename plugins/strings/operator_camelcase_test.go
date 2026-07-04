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

package rostrings

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
	"golang.org/x/text/language"
)

func TestCamelCase(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	for _, t := range allCaseTests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(t.input),
				CamelCase[string](),
			),
		)
		is.Equal([]string{t.output.CamelCase}, values)
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Empty[string](),
				CamelCase[string](),
			),
		)
		is.Equal([]string{}, values)
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Throw[string](assert.AnError),
				CamelCase[string](),
			),
		)
		is.Equal([]string{}, values)
		is.EqualError(err, assert.AnError.Error())
	}
}

func TestCamelCaseWithLanguage(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tests := []struct {
		input string
		tag   language.Tag
		want  string
	}{
		{"hello world", language.English, "helloWorld"},
		// Turkish: first word 'I' lowercases to 'ı' (U+0131); second word 'c' title-cases to 'C'
		{"Istanbul city", language.Turkish, "ıstanbulCity"},
	}

	for _, tc := range tests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(tc.input),
				CamelCaseWithLanguage[string](tc.tag),
			),
		)
		is.Equal([]string{tc.want}, values)
		is.NoError(err)
	}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Empty[string](),
			CamelCaseWithLanguage[string](language.English),
		),
	)
	is.Equal([]string{}, values)
	is.NoError(err)

	values, err = ro.Collect(
		ro.Pipe1(
			ro.Throw[string](assert.AnError),
			CamelCaseWithLanguage[string](language.English),
		),
	)
	is.Equal([]string{}, values)
	is.EqualError(err, assert.AnError.Error())
}
