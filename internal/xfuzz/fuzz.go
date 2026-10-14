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

// Package xfuzz holds helpers shared by the fuzz tests of the whole repository.
package xfuzz

import (
	"os"
	"strconv"
	"testing"
)

const (
	// FuzzIterationsEnv is the environment variable that overrides the number of
	// iterations of every fuzz target. `make fuzz` also forwards it to `-fuzztime`.
	FuzzIterationsEnv = "RO_FUZZ_ITERATIONS"

	// defaultFuzzIterations is enough interleavings to catch most races locally
	// in a few seconds, without slowing down `make test`.
	defaultFuzzIterations = 100

	// shortFuzzIterations keeps `go test -short` fast while still exercising
	// several interleavings per target.
	shortFuzzIterations = 10
)

// FuzzIterations returns the number of iterations shared by every fuzz target.
//
// An explicit RO_FUZZ_ITERATIONS wins over everything else. Otherwise, the
// count is reduced under `go test -short`. Invalid or non-positive values are
// ignored.
func FuzzIterations() int {
	if raw, ok := os.LookupEnv(FuzzIterationsEnv); ok {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}

	if testing.Short() {
		return shortFuzzIterations
	}

	return defaultFuzzIterations
}

// AddSeeds registers FuzzIterations() seeds in the fuzz corpus, so a plain
// `go test -race` explores that many interleavings without the -fuzz flag.
//
// gen must be deterministic: it receives the seed index and returns the
// arguments of f.Add. The arguments must match the fuzz function signature.
func AddSeeds(f *testing.F, gen func(i int) []any) {
	f.Helper()

	for i := 0; i < FuzzIterations(); i++ {
		f.Add(gen(i)...)
	}
}
