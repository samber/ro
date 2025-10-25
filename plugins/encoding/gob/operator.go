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


package rogob

import (
	"bytes"
	"encoding/gob"

	"github.com/samber/ro"
)

// Encode encodes values to gob binary format.
// Play: https://go.dev/play/p/HdU4DMTagoA
func Encode[T any]() func(ro.Observable[T]) ro.Observable[[]byte] {
	return ro.MapErr(func(v T) ([]byte, error) {
		var writer bytes.Buffer
		err := gob.NewEncoder(&writer).Encode(v)
		return writer.Bytes(), err
	})
}

// Decode decodes gob binary data to values.
// Play: https://go.dev/play/p/cH3AiWEwFQe
func Decode[T any]() func(ro.Observable[[]byte]) ro.Observable[T] {
	return ro.MapErr(func(v []byte) (T, error) {
		var output T
		buf := bytes.NewBuffer(v)
		err := gob.NewDecoder(buf).Decode(&output)
		return output, err
	})
}
