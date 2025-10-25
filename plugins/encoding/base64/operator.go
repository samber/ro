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


package robase64

import (
	"encoding/base64"

	"github.com/samber/ro"
)

// Encode encodes the input into a base64 string.
//
// Example:
//
//	ro.Pipe1(
//		ro.Just([]byte("hello")),
//		robase64.Encode(base64.StdEncoding),
//	)
// Play: https://go.dev/play/p/PZCXxLxn5AF
func Encode[T ~[]byte](encoder *base64.Encoding) func(ro.Observable[T]) ro.Observable[string] {
	return ro.Map(func(v T) string {
		return encoder.EncodeToString([]byte(v))
	})
}

// Decode decodes the input from a base64 string.
//
// Example:
//
//	ro.Pipe1(
//		ro.Just("aGVsbG8="),
//		robase64.Decode(base64.StdEncoding),
//	)
// Play: https://go.dev/play/p/dTPmEzSHgi7
func Decode[T ~string](encoder *base64.Encoding) func(ro.Observable[T]) ro.Observable[[]byte] {
	return ro.MapErr(func(v T) ([]byte, error) {
		return encoder.DecodeString(string(v))
	})
}
