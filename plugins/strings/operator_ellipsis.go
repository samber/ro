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
)

func ellipsis(str string, length int) string {
	str = strings.TrimSpace(str)

	if len(str) > length {
		if len(str) < 3 || length < 3 {
			return "..."
		}
		return strings.TrimSpace(str[0:length-3]) + "..."
	}

	return str
}

// Ellipsis truncates the string to the specified length and appends "..." if the string is longer than the specified length.
// Play: https://go.dev/play/p/Yqx1pUep0uX
func Ellipsis[T ~string](length int) func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(ellipsis(string(value), length))
		},
	)
}
