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

func FuzzSCContains(f *testing.F) {
	f.Skip("race: contains-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "Contains", seed, mask, n, func(source ro.Observable[int]) ro.Observable[bool] {
			// >= makes every item after the decision a match too, so a re-fired decision is visible.
			if variant == 0 {
				return ro.Contains(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return ro.ContainsI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "Contains", sink, probe, []bool{k < n}, false, mask)
	})
}

func FuzzSCFind(f *testing.F) {
	f.Skip("race: find-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "Find", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			if variant == 0 {
				return ro.Find(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return ro.FindI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		fuzzSCCheck(t, "Find", sink, probe, fuzzSCWantFirst(n, k), false, mask)
	})
}

// FuzzSCAll is the control: All short-circuits correctly since #429.
func FuzzSCAll(f *testing.F) {
	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := fuzzSCScenario(seed)
		probe := &fuzzSCProbe{}

		sink := fuzzSCRunRaw(t, "All", seed, mask, n, func(source ro.Observable[int]) ro.Observable[bool] {
			if variant == 0 {
				return ro.All(func(v int) bool { probe.call(v >= k); return v < k })(source)
			}

			return ro.AllI(func(v int, _ int64) bool { probe.call(v >= k); return v < k })(source)
		})

		fuzzSCCheck(t, "All", sink, probe, []bool{k >= n}, false, mask)
	})
}
