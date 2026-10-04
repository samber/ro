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

package xtest

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFuzzIterations(t *testing.T) {
	is := assert.New(t)

	fallback := defaultFuzzIterations
	if testing.Short() {
		fallback = shortFuzzIterations
	}

	t.Setenv(FuzzIterationsEnv, "")
	is.Equal(fallback, FuzzIterations())

	t.Setenv(FuzzIterationsEnv, "2500")
	is.Equal(2500, FuzzIterations())

	// Pivot values: invalid, zero and negative fall back to the default.
	for _, raw := range []string{"abc", "0", "-1", "1.5"} {
		t.Setenv(FuzzIterationsEnv, raw)
		is.Equal(fallback, FuzzIterations(), raw)
	}
}

func FuzzAddSeeds(f *testing.F) {
	count := 0

	AddSeeds(f, func(i int) []any {
		count++
		return []any{int64(i)}
	})

	if count != FuzzIterations() {
		f.Fatalf("registered %d seeds, want %d", count, FuzzIterations())
	}

	f.Fuzz(func(t *testing.T, seed int64) {
		if seed < 0 {
			t.Fatalf("negative seed %d", seed)
		}
	})
}
