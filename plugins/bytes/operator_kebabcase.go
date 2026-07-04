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
	"bytes"

	"github.com/samber/ro"
	"golang.org/x/text/language"
)

func kebabCase(str []byte) []byte {
	items := words(str)
	for i := range items {
		items[i] = lowerEnglish(items[i])
	}
	return bytes.Join(items, []byte("-"))
}

func kebabCaseWithLanguage(str []byte, tag language.Tag) []byte {
	items := words(str)
	if len(items) == 0 {
		return []byte{}
	}
	pool, c := acquireLowerCaser(tag)
	defer pool.Put(c)
	for i := range items {
		items[i] = c.Bytes(items[i])
	}
	return bytes.Join(items, []byte("-"))
}

// KebabCase converts the string to kebab case.
// Play: https://go.dev/play/p/CeGTAyeZu2W
func KebabCase[T ~[]byte]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(kebabCase(value))
		},
	)
}

// KebabCaseWithLanguage converts the byte slice to kebab case using locale-aware lowercasing.
func KebabCaseWithLanguage[T ~[]byte](tag language.Tag) func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(kebabCaseWithLanguage(value, tag))
		},
	)
}
