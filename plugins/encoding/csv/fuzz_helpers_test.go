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

package rocsv

import (
	"os"
	"strconv"
	"testing"
)

const (
	// fuzzIterationsEnv overrides the number of seeds of every fuzz target.
	fuzzIterationsEnv = "RO_FUZZ_ITERATIONS"

	// defaultFuzzIterations is enough interleavings to catch most races locally without slowing down plain test runs.
	defaultFuzzIterations = 100

	// shortFuzzIterations keeps `go test -short` fast.
	shortFuzzIterations = 10
)

// fuzzIterations mirrors internal/xtest.FuzzIterations, which plugin modules cannot import.
func fuzzIterations() int {
	if raw, ok := os.LookupEnv(fuzzIterationsEnv); ok {
		if n, err := strconv.Atoi(raw); err == nil && n > 0 {
			return n
		}
	}

	if testing.Short() {
		return shortFuzzIterations
	}

	return defaultFuzzIterations
}

// addSeeds registers fuzzIterations() deterministic seeds so a plain `go test` explores that many scenarios.
func addSeeds(f *testing.F, gen func(i int) []any) {
	f.Helper()

	for i := 0; i < fuzzIterations(); i++ {
		f.Add(gen(i)...)
	}
}
