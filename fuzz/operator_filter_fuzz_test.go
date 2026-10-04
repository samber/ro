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

func intRange(from, to int) []int {
	out := []int{}
	for i := from; i < to; i++ {
		out = append(out, i)
	}

	return out
}

func FuzzFirst(f *testing.F) {
	f.Skip("race: first-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := decodeShortCircuitScenario(seed)
		probe := &predicateCallCounter{}

		sink := runWithRawSink(t, "First", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			if variant == 0 {
				return ro.First(func(v int) bool { probe.call(v >= k); return v >= k })(source)
			}

			return ro.FirstI(func(v int, _ int64) bool { probe.call(v >= k); return v >= k })(source)
		})

		checkSingleResult(t, "First", sink, probe, expectedItemAt(n, k), k >= n, mask)
	})
}

func FuzzElementAt(f *testing.F) {
	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := decodeShortCircuitScenario(seed)

		sink := runWithRawSink(t, "ElementAt", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.ElementAt[int](k)(source)
		})

		checkSingleResult(t, "ElementAt", sink, nil, expectedItemAt(n, k), k >= n, mask)
	})
}

func FuzzElementAtOrDefault(f *testing.F) {
	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := decodeShortCircuitScenario(seed)

		sink := runWithRawSink(t, "ElementAtOrDefault", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.ElementAtOrDefault(int64(k), unreachableDefault)(source)
		})

		want := []int{unreachableDefault}
		if k < n {
			want = []int{k}
		}

		checkSingleResult(t, "ElementAtOrDefault", sink, nil, want, false, mask)
	})
}

func FuzzHead(f *testing.F) {
	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, _, _ := decodeShortCircuitScenario(seed)

		sink := runWithRawSink(t, "Head", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.Head[int]()(source)
		})

		want := []int{}
		if n > 0 {
			want = []int{0}
		}

		checkSingleResult(t, "Head", sink, nil, want, n == 0, mask)
	})
}

func FuzzTake(f *testing.F) {
	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, _ := decodeShortCircuitScenario(seed)

		sink := runWithRawSink(t, "Take", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			return ro.Take[int](int64(k))(source)
		})

		checkSingleResult(t, "Take", sink, nil, intRange(0, minInt(k, n)), false, mask)
	})
}

func FuzzTakeWhile(f *testing.F) {
	addShortCircuitSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		n, k, variant := decodeShortCircuitScenario(seed)
		probe := &predicateCallCounter{}

		sink := runWithRawSink(t, "TakeWhile", seed, mask, n, func(source ro.Observable[int]) ro.Observable[int] {
			if variant == 0 {
				return ro.TakeWhile(func(v int) bool { probe.call(v >= k); return v < k })(source)
			}

			return ro.TakeWhileI(func(v int, _ int64) bool { probe.call(v >= k); return v < k })(source)
		})

		checkSingleResult(t, "TakeWhile", sink, probe, intRange(0, minInt(k, n)), false, mask)
	})
}
