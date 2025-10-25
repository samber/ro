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
)

func snakeCase(str []byte) []byte {
	items := words(str)
	for i := range items {
		items[i] = bytes.ToLower(items[i])
	}
	return bytes.Join(items, []byte{'_'})
}

// SnakeCase converts the string to snake case.
// Play: https://go.dev/play/p/Cf5GQNC9KpW
func SnakeCase[T ~[]byte]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(snakeCase(value))
		},
	)
}
