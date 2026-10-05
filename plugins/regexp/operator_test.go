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

package roregexp

import (
	"regexp"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestFind(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := Find[[]byte](pattern)

	observable := ro.FromSlice([][]byte{
		[]byte("abc123def"),
		[]byte("no numbers"),
		[]byte("456xyz789"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]byte{
		[]byte("123"),
		nil, // no match
		[]byte("456"),
	}

	assert.Equal(t, expected, result)
}

func TestFindString(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := FindString[string](pattern)

	observable := ro.FromSlice([]string{
		"abc123def",
		"no numbers",
		"456xyz789",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := []string{
		"123",
		"", // no match
		"456",
	}

	assert.Equal(t, expected, result)
}

func TestFindSubmatch(t *testing.T) {
	pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
	operator := FindSubmatch[[]byte](pattern)

	observable := ro.FromSlice([][]byte{
		[]byte("123abc456def"),
		[]byte("no match"),
		[]byte("789xyz"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][][]byte{
		{[]byte("123abc"), []byte("123"), []byte("abc")},
		nil, // no match
		{[]byte("789xyz"), []byte("789"), []byte("xyz")},
	}

	assert.Equal(t, expected, result)
}

func TestFindStringSubmatch(t *testing.T) {
	pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
	operator := FindStringSubmatch[string](pattern)

	observable := ro.FromSlice([]string{
		"123abc456def",
		"no match",
		"789xyz",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]string{
		{"123abc", "123", "abc"},
		nil, // no match
		{"789xyz", "789", "xyz"},
	}

	assert.Equal(t, expected, result)
}

func TestFindAll(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := FindAll[[]byte](pattern, -1)

	observable := ro.FromSlice([][]byte{
		[]byte("abc123def456ghi"),
		[]byte("no numbers"),
		[]byte("789xyz123"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][][]byte{
		{[]byte("123"), []byte("456")},
		nil, // no match
		{[]byte("789"), []byte("123")},
	}

	assert.Equal(t, expected, result)

	t.Run("with limit", func(t *testing.T) {
		pattern := regexp.MustCompile(`\d+`)
		operator := FindAll[[]byte](pattern, 2) // limit to 2 matches

		observable := ro.FromSlice([][]byte{
			[]byte("abc123def456ghi789"),
			[]byte("no numbers"),
			[]byte("789xyz123abc456"),
		})

		result, err := ro.Collect(operator(observable))
		if err != nil {
			t.Fatal(err)
		}

		expected := [][][]byte{
			{[]byte("123"), []byte("456")}, // only first 2 matches
			nil,                            // no match
			{[]byte("789"), []byte("123")}, // only first 2 matches
		}

		assert.Equal(t, expected, result)
	})
}

func TestFindAllString(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := FindAllString[string](pattern, -1)

	observable := ro.FromSlice([]string{
		"abc123def456ghi",
		"no numbers",
		"789xyz123",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]string{
		{"123", "456"},
		nil, // no match
		{"789", "123"},
	}

	assert.Equal(t, expected, result)

	t.Run("with limit", func(t *testing.T) {
		pattern := regexp.MustCompile(`\d+`)
		operator := FindAllString[string](pattern, 1) // limit to 1 match

		observable := ro.FromSlice([]string{
			"abc123def456ghi789",
			"no numbers",
			"789xyz123abc456",
		})

		result, err := ro.Collect(operator(observable))
		if err != nil {
			t.Fatal(err)
		}

		expected := [][]string{
			{"123"}, // only first match
			nil,     // no match
			{"789"}, // only first match
		}

		assert.Equal(t, expected, result)
	})
}

func TestFindAllSubmatch(t *testing.T) {
	pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
	operator := FindAllSubmatch[[]byte](pattern, -1)

	observable := ro.FromSlice([][]byte{
		[]byte("123abc456def789ghi"),
		[]byte("no match"),
		[]byte("789xyz123abc"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][][][]byte{
		{
			{[]byte("123abc"), []byte("123"), []byte("abc")},
			{[]byte("456def"), []byte("456"), []byte("def")},
			{[]byte("789ghi"), []byte("789"), []byte("ghi")},
		},
		nil, // no match
		{
			{[]byte("789xyz"), []byte("789"), []byte("xyz")},
			{[]byte("123abc"), []byte("123"), []byte("abc")},
		},
	}

	assert.Equal(t, expected, result)

	t.Run("with limit", func(t *testing.T) {
		pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
		operator := FindAllSubmatch[[]byte](pattern, 2) // limit to 2 matches

		observable := ro.FromSlice([][]byte{
			[]byte("123abc456def789ghi"),
			[]byte("no match"),
			[]byte("789xyz123abc456def"),
		})

		result, err := ro.Collect(operator(observable))
		if err != nil {
			t.Fatal(err)
		}

		expected := [][][][]byte{
			{
				{[]byte("123abc"), []byte("123"), []byte("abc")},
				{[]byte("456def"), []byte("456"), []byte("def")},
			},
			nil, // no match
			{
				{[]byte("789xyz"), []byte("789"), []byte("xyz")},
				{[]byte("123abc"), []byte("123"), []byte("abc")},
			},
		}

		assert.Equal(t, expected, result)
	})
}

func TestFindAllStringSubmatch(t *testing.T) {
	pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
	operator := FindAllStringSubmatch[string](pattern, -1)

	observable := ro.FromSlice([]string{
		"123abc456def789ghi",
		"no match",
		"789xyz123abc",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][][]string{
		{
			{"123abc", "123", "abc"},
			{"456def", "456", "def"},
			{"789ghi", "789", "ghi"},
		},
		nil, // no match
		{
			{"789xyz", "789", "xyz"},
			{"123abc", "123", "abc"},
		},
	}

	assert.Equal(t, expected, result)

	t.Run("with limit", func(t *testing.T) {
		pattern := regexp.MustCompile(`(\d+)([a-z]+)`)
		operator := FindAllStringSubmatch[string](pattern, 1) // limit to 1 match

		observable := ro.FromSlice([]string{
			"123abc456def789ghi",
			"no match",
			"789xyz123abc456def",
		})

		result, err := ro.Collect(operator(observable))
		if err != nil {
			t.Fatal(err)
		}

		expected := [][][]string{
			{
				{"123abc", "123", "abc"},
			},
			nil, // no match
			{
				{"789xyz", "789", "xyz"},
			},
		}

		assert.Equal(t, expected, result)
	})
}

func TestMatch(t *testing.T) {
	pattern := regexp.MustCompile(`^\d+$`)
	operator := Match[[]byte](pattern)

	observable := ro.FromSlice([][]byte{
		[]byte("123"),
		[]byte("abc"),
		[]byte("456"),
		[]byte("123abc"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := []bool{
		true,  // "123" matches
		false, // "abc" doesn't match
		true,  // "456" matches
		false, // "123abc" doesn't match
	}

	assert.Equal(t, expected, result)
}

func TestMatchString(t *testing.T) {
	pattern := regexp.MustCompile(`^\d+$`)
	operator := MatchString[string](pattern)

	observable := ro.FromSlice([]string{
		"123",
		"abc",
		"456",
		"123abc",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := []bool{
		true,  // "123" matches
		false, // "abc" doesn't match
		true,  // "456" matches
		false, // "123abc" doesn't match
	}

	assert.Equal(t, expected, result)
}

func TestReplaceAll(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := ReplaceAll[[]byte](pattern, []byte("NUMBER"))

	observable := ro.FromSlice([][]byte{
		[]byte("abc123def456"),
		[]byte("no numbers"),
		[]byte("789xyz"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]byte{
		[]byte("abcNUMBERdefNUMBER"),
		[]byte("no numbers"),
		[]byte("NUMBERxyz"),
	}

	assert.Equal(t, expected, result)
}

func TestReplaceAllString(t *testing.T) {
	pattern := regexp.MustCompile(`\d+`)
	operator := ReplaceAllString[string](pattern, "NUMBER")

	observable := ro.FromSlice([]string{
		"abc123def456",
		"no numbers",
		"789xyz",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := []string{
		"abcNUMBERdefNUMBER",
		"no numbers",
		"NUMBERxyz",
	}

	assert.Equal(t, expected, result)
}

func TestFilterMatch(t *testing.T) {
	pattern := regexp.MustCompile(`^\d+$`)
	operator := FilterMatch[[]byte](pattern)

	observable := ro.FromSlice([][]byte{
		[]byte("123"),
		[]byte("abc"),
		[]byte("456"),
		[]byte("123abc"),
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := [][]byte{
		[]byte("123"),
		[]byte("456"),
	}

	assert.Equal(t, expected, result)
}

func TestFilterMatchString(t *testing.T) {
	pattern := regexp.MustCompile(`^\d+$`)
	operator := FilterMatchString[string](pattern)

	observable := ro.FromSlice([]string{
		"123",
		"abc",
		"456",
		"123abc",
	})

	result, err := ro.Collect(operator(observable))
	if err != nil {
		t.Fatal(err)
	}

	expected := []string{
		"123",
		"456",
	}

	assert.Equal(t, expected, result)
}
