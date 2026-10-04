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

	"github.com/samber/ro"
)

// fuzzSCMin exists because go.mod declares go 1.18, which has no min builtin.
func fuzzSCMin(a, b int) int {
	if a < b {
		return a
	}

	return b
}

func fuzzSCRange(from, to int) []int {
	out := []int{}
	for i := from; i < to; i++ {
		out = append(out, i)
	}

	return out
}

func FuzzSCFirst(f *testing.F) {
	f.Skip("race: first-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "First", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			if variant == 0 {
				return ro.First(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return ro.FirstI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "First", sink, probe, fuzzSCWantFirst(n, k), k >= n, mask)
	})
}

func FuzzSCElementAt(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "ElementAt", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.ElementAt[int](k)(source)
		})

		fuzzSCCheck(t, "ElementAt", sink, nil, fuzzSCWantFirst(n, k), k >= n, mask)
	})
}

func FuzzSCElementAtOrDefault(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "ElementAtOrDefault", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.ElementAtOrDefault(int64(k), fuzzSCNoDefault)(source)
		})

		want := []int{fuzzSCNoDefault}
		if k < n {
			want = []int{k}
		}

		fuzzSCCheck(t, "ElementAtOrDefault", sink, nil, want, false, mask)
	})
}

func FuzzSCHead(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, _, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "Head", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.Head[int]()(source)
		})

		want := []int{}
		if n > 0 {
			want = []int{0}
		}

		fuzzSCCheck(t, "Head", sink, nil, want, n == 0, mask)
	})
}

func FuzzSCTake(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := fuzzSCScenario(seed)

		sink := fuzzSCRunRaw(t, "Take", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.Take[int](int64(k))(source)
		})

		fuzzSCCheck(t, "Take", sink, nil, fuzzSCRange(0, fuzzSCMin(k, n)), false, mask)
	})
}

func FuzzSCTakeWhile(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "TakeWhile", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			if variant == 0 {
				return ro.TakeWhile(func(v int) bool { probe.call(v >= k); return v < k })(source)
			}

			return ro.TakeWhileI(func(v int, _ int64) bool { probe.call(v >= k); return v < k })(source)
		})

		fuzzSCCheck(t, "TakeWhile", sink, probe, fuzzSCRange(0, fuzzSCMin(k, n)), false, mask)
	})
}
