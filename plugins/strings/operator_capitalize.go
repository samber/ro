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
	"github.com/samber/ro"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

func capitalize(str string) string {
	return cases.Title(language.English).String(str)
}

// Capitalize capitalizes the first letter of the string.
// Play: https://go.dev/play/p/7hK8m9jL3nS
func Capitalize[T ~string]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(capitalize(string(value)))
		},
	)
}
