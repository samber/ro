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
	"strings"

	"github.com/samber/ro"
	"golang.org/x/text/language"
)

func toCamelCase(str string) string {
	items := words(str)
	for i, item := range items {
		item = strings.ToLower(item)
		if i > 0 {
			item = capitalize(item)
		}
		items[i] = item
	}
	return strings.Join(items, "")
}

func toCamelCaseWithLanguage(str string, tag language.Tag) string {
	items := words(str)
	if len(items) == 0 {
		return ""
	}
	lPool, lc := acquireLowerCaser(tag)
	tPool, tc := acquireTitleCaser(tag)
	defer lPool.Put(lc)
	defer tPool.Put(tc)
	items[0] = lc.String(items[0])
	for i := 1; i < len(items); i++ {
		items[i] = tc.String(items[i])
	}
	return strings.Join(items, "")
}

// CamelCase converts the string to camel case.
// Play: https://go.dev/play/p/MMmhpwApG1y
func CamelCase[T ~string]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(toCamelCase(string(value)))
		},
	)
}

// CamelCaseWithLanguage converts the string to camel case using locale-aware casing.
func CamelCaseWithLanguage[T ~string](tag language.Tag) func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(toCamelCaseWithLanguage(string(value), tag))
		},
	)
}
