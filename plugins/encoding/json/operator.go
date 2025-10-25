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


package rojson

import (
	"encoding/json"

	"github.com/samber/ro"
)

// Marshal encodes values to JSON format.
// Play: https://go.dev/play/p/8hN7m8kK4jT
func Marshal[T any]() func(ro.Observable[T]) ro.Observable[[]byte] {
	return ro.MapErr(func(v T) ([]byte, error) {
		return json.Marshal(v)
	})
}

// Unmarshal decodes JSON data to values.
// Play: https://go.dev/play/p/8hN7m8kK4jT
func Unmarshal[T any]() func(ro.Observable[[]byte]) ro.Observable[T] {
	return ro.MapErr(func(v []byte) (T, error) {
		var t T
		err := json.Unmarshal(v, &t)
		return t, err
	})
}
