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

package fuzz

import (
	"testing"

	"github.com/samber/ro/internal/xfuzz"
)

// maxItems bounds the number of items a fuzz input can request, to keep one iteration fast.
const maxItems = 64

// integer is every integer type a fuzz input can use.
type integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

// fuzzSeeds registers one seed per iteration (RO_FUZZ_ITERATIONS, 100 by default), so that a plain
// `go test -race` explores them without -fuzz. generate must be deterministic and return the
// fuzz arguments in the order of the f.Fuzz callback.
func fuzzSeeds(f *testing.F, generate func(i int) []any) {
	f.Helper()

	xfuzz.AddSeeds(f, generate)
}

// bounded maps any fuzz input, negative or huge ones included, into [low, high].
func bounded[T integer](value T, low, high int) int {
	span := int64(high-low) + 1

	remainder := int64(value) % span
	if remainder < 0 {
		remainder = -remainder
	}

	return low + int(remainder)
}

// seedStrides are odd multipliers, one per argument position, that spread the seed index over a whole
// byte range and keep the arguments of one seed from moving in lockstep.
var seedStrides = [...]int{37, 53, 29, 41, 61}

// seedByte returns the value of the argumentIndex-th uint8 argument of seed number i.
func seedByte(i, argumentIndex int) uint8 {
	return uint8(i * seedStrides[argumentIndex%len(seedStrides)])
}
