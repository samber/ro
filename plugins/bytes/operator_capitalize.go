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
	"sync"

	"github.com/samber/ro"
	"golang.org/x/text/cases"
	"golang.org/x/text/language"
)

// titleCaserPool reuses cases.Title casers: constructing one is far more
// expensive than the casing itself, and a Caser is not safe for concurrent
// use, so it cannot be a plain package-level singleton.
var titleCaserPool = sync.Pool{
	New: func() any {
		c := cases.Title(language.English)
		return &c
	},
}

func capitalize(str []byte) []byte {
	c, _ := titleCaserPool.Get().(*cases.Caser) // Pool.New always returns *cases.Caser, so the assertion never fails.
	defer titleCaserPool.Put(c)
	return c.Bytes(str)
}

// Capitalize capitalizes the first letter of the string.
// Play: https://go.dev/play/p/gAKIElJIUun
func Capitalize[T ~[]byte]() func(destination ro.Observable[T]) ro.Observable[T] {
	return ro.Map(
		func(value T) T {
			return T(capitalize(value))
		},
	)
}
