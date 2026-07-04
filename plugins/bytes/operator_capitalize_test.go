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

package robytes

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
	"golang.org/x/text/language"
)

func TestCapitalize(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tests := []struct {
		input string
		want  string
	}{
		{"", ""},
		{"hello", "Hello"},
		{"heLLO", "Hello"},
		{" hello", " Hello"},
		{"hello ", "Hello "},
		{" hello ", " Hello "},
		{"123test", "123Test"},
		{"test-case", "Test-Case"},
		{"multi word string", "Multi Word String"},
		{"already Capitalized", "Already Capitalized"},
	}

	for _, t := range tests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just([]byte(t.input)),
				Capitalize[[]byte](),
			),
		)
		is.Equal([]byte(t.want), values[0])
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Empty[[]byte](),
				Capitalize[[]byte](),
			),
		)
		is.Equal([][]byte{}, values)
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Throw[[]byte](assert.AnError),
				Capitalize[[]byte](),
			),
		)
		is.Equal([][]byte{}, values)
		is.EqualError(err, assert.AnError.Error())
	}
}

func TestCapitalizeWithLanguage(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tests := []struct {
		input []byte
		tag   language.Tag
		want  []byte
	}{
		{[]byte("hello"), language.English, []byte("Hello")},
		{[]byte("heLLO"), language.English, []byte("Hello")},
		// Turkish: 'i' title-cases to 'İ' (U+0130)
		{[]byte("istanbul"), language.Turkish, []byte("İstanbul")},
	}

	for _, tc := range tests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(tc.input),
				CapitalizeWithLanguage[[]byte](tc.tag),
			),
		)
		is.Equal(tc.want, values[0])
		is.NoError(err)
	}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Empty[[]byte](),
			CapitalizeWithLanguage[[]byte](language.English),
		),
	)
	is.Equal([][]byte{}, values)
	is.NoError(err)

	values, err = ro.Collect(
		ro.Pipe1(
			ro.Throw[[]byte](assert.AnError),
			CapitalizeWithLanguage[[]byte](language.English),
		),
	)
	is.Equal([][]byte{}, values)
	is.EqualError(err, assert.AnError.Error())
}
