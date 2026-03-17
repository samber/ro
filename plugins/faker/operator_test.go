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

package rofaker

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type person struct {
	Name  string `faker:"name"`
	Email string `faker:"email"`
}

func TestFake(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Fake[person](3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v.Name)
		is.NotEmpty(v.Email)
	}

	// zero count emits no items
	values, err = ro.Collect(Fake[person](0))
	is.NoError(err)
	is.Empty(values)
}

func TestName(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Name(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}

	values, err = ro.Collect(Name(0))
	is.NoError(err)
	is.Empty(values)
}

func TestFirstName(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(FirstName(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestLastName(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(LastName(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestEmail(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Email(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
		is.Contains(v, "@")
	}
}

func TestPhoneNumber(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(PhoneNumber(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestURL(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(URL(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestIPv4(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(IPv4(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestIPv6(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(IPv6(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestUUIDHyphenated(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(UUIDHyphenated(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
		is.Contains(v, "-")
	}
}

func TestUUIDDigit(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(UUIDDigit(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
		is.NotContains(v, "-")
	}
}

func TestWord(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Word(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestSentence(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Sentence(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestParagraph(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Paragraph(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.NotEmpty(v)
	}
}

func TestLatitude(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Latitude(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.GreaterOrEqual(v, -90.0)
		is.LessOrEqual(v, 90.0)
	}
}

func TestLongitude(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(Longitude(3))
	is.NoError(err)
	is.Len(values, 3)

	for _, v := range values {
		is.GreaterOrEqual(v, -180.0)
		is.LessOrEqual(v, 180.0)
	}
}
