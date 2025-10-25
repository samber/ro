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
	"unicode"

	"github.com/samber/ro"
)

func words(str []byte) [][]byte {
	str = splitWordReg.ReplaceAll(str, []byte(`$1$3$5$7 $2$4$6$8$9`))
	// example: Int8Value => Int 8Value => Int 8 Value
	str = splitNumberLetterReg.ReplaceAll(str, []byte("$1 $2"))
	var result bytes.Buffer
	for _, r := range str {
		if unicode.IsLetter(rune(r)) || unicode.IsDigit(rune(r)) {
			result.WriteByte(r)
		} else {
			result.WriteByte(' ')
		}
	}
	return bytes.Fields(result.Bytes())
}

// Words splits the string into words.
// Play: https://go.dev/play/p/N6fiwqBry5e
func Words[T ~[]byte]() func(destination ro.Observable[T]) ro.Observable[[]T] {
	return ro.Map(
		func(value T) []T {
			output := make([]T, 0)
			for _, word := range words(value) {
				output = append(output, T(word))
			}
			return output
		},
	)
}
