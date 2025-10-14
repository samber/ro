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
)

func TestEllipsis(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tests := []struct {
		input  string
		length int
		want   string
	}{
		{"12345", 5, "12345"},
		{"12345", 4, "1..."},
		{"	12345  ", 4, "1..."},
		{"12345", 6, "12345"},
		{"12345", 10, "12345"},
		{"  12345  ", 10, "12345"},
		{"12345", 3, "..."},
		{"12345", 2, "..."},
		{"12345", -1, "..."},
		{" hello   world ", 9, "hello..."},
	}

	for _, t := range tests {
		values, err := ro.Collect(
			ro.Pipe1(
				ro.Just([]byte(t.input)),
				Ellipsis[[]byte](t.length),
			),
		)
		is.Equal([]byte(t.want), values[0])
		is.Nil(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Empty[[]byte](),
				Ellipsis[[]byte](42),
			),
		)
		is.Equal([][]byte{}, values)
		is.Nil(err)

		values, err = ro.Collect(
			ro.Pipe1(
				ro.Throw[[]byte](assert.AnError),
				Ellipsis[[]byte](42),
			),
		)
		is.Equal([][]byte{}, values)
		is.EqualError(err, assert.AnError.Error())
	}
}
