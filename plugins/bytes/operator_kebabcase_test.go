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

func TestKebabCase(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	for _, t := range allCaseTests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just([]byte(t.input)),
				KebabCase[[]byte](),
			),
		)
		is.Equal([]byte(t.output.KebabCase), values[0])
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Empty[[]byte](),
				KebabCase[[]byte](),
			),
		)
		is.Equal([][]byte{}, values)
		is.NoError(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Throw[[]byte](assert.AnError),
				KebabCase[[]byte](),
			),
		)
		is.Equal([][]byte{}, values)
		is.EqualError(err, assert.AnError.Error())
	}
}

func TestKebabCaseWithLanguage(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tests := []struct {
		input []byte
		tag   language.Tag
		want  []byte
	}{
		{[]byte("HelloWorld"), language.English, []byte("hello-world")},
		// Turkish: 'I' lowercases to 'ı' (U+0131)
		{[]byte("IstanbulCity"), language.Turkish, []byte("ıstanbul-city")},
	}

	for _, tc := range tests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(tc.input),
				KebabCaseWithLanguage[[]byte](tc.tag),
			),
		)
		is.Equal(tc.want, values[0])
		is.NoError(err)
	}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Empty[[]byte](),
			KebabCaseWithLanguage[[]byte](language.English),
		),
	)
	is.Equal([][]byte{}, values)
	is.NoError(err)

	values, err = ro.Collect(
		ro.Pipe1(
			ro.Throw[[]byte](assert.AnError),
			KebabCaseWithLanguage[[]byte](language.English),
		),
	)
	is.Equal([][]byte{}, values)
	is.EqualError(err, assert.AnError.Error())
}
