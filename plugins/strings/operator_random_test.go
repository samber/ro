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
	"math/rand"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestRandom(t *testing.T) {
	t.Parallel()

	rand.Seed(time.Now().UnixNano())

	t.Run("distinct outputs from lowercase charset", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(struct{}{}),
				Random[struct{}](100, LowerCaseLettersCharset),
			),
		)
		is.Equal(100, utf8.RuneCountInString(values[0]))
		is.Subset(LowerCaseLettersCharset, []rune(values[0]))
		is.Nil(err)

		values2, err := ro.Collect(
			ro.Pipe1(
				ro.Just(struct{}{}),
				Random[struct{}](100, LowerCaseLettersCharset),
			),
		)
		is.Equal(100, utf8.RuneCountInString(values2[0]))
		is.Subset(LowerCaseLettersCharset, []rune(values2[0]))
		is.NotEqual(values, values2)
		is.Nil(err)
	})

	t.Run("multibyte charset", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just(struct{}{}),
				Random[struct{}](100, []rune("明1好休2林森")),
			),
		)
		is.Equal(100, utf8.RuneCountInString(values[0]))
		is.Subset([]rune("明1好休2林森"), []rune(values[0]))
		is.Nil(err)
	})

	t.Run("single-char charset does not divide by zero", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		values3, err := ro.Collect(
			ro.Pipe1(
				ro.Just(struct{}{}),
				Random[struct{}](10, []rune{65}),
			),
		)
		is.Equal(10, utf8.RuneCountInString(values3[0]))
		is.Equal("AAAAAAAAAA", values3[0])
		is.Nil(err)
	})

	t.Run("panics on empty charset", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithError(
			"rostrings.Random: charset must not be empty",
			func() {
				_ = Random[struct{}](100, []rune{})
			},
		)
	})

	t.Run("panics on zero size", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithError(
			"rostrings.Random: size must be greater than 0",
			func() {
				_ = Random[struct{}](0, LowerCaseLettersCharset)
			},
		)
	})
}
