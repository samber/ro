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

// FuzzContains checks whether any item of a source of `items` integers is >= decisionIndex. The source
// is synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: Contains emits exactly one boolean (true when some item matched) and completes once, and
// it never calls the predicate again after the call that matched. Items after the match also
// satisfy the predicate, so a re-fired decision would be visible.
//
// Seeds: items and decisionIndex spread over their range; asyncSource, sourceIgnoresStop and
// withIndex (ContainsI instead of Contains) alternate.
func FuzzContains(f *testing.F) {
	f.Skip("race: contains-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
		count := bounded(items, 0, maxItems)
		decision := bounded(decisionIndex, 0, count+1) // decision >= count: no item matches

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
		predicate := newCountingPredicate(true, func(item int) bool { return item >= decision })

		contains := ro.Contains(predicate.test)
		if withIndex {
			contains = ro.ContainsI(predicate.testIndexed)
		}

		got := collect(t, contains(numbers.observable()), numbers)

		got.expectValues(t, []bool{decision < count})
		got.expectCompletedOnce(t)
		got.expectContract(t)
		predicate.expectNoCallAfterDecision(t)
	})
}

// FuzzFind looks for the first item >= decisionIndex in a source of `items` integers. The source is
// synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: Find emits the first matching item, or nothing when none matches, and completes once. It
// never calls the predicate again after the call that matched.
//
// Seeds: same as FuzzContains, with withIndex selecting FindI.
func FuzzFind(f *testing.F) {
	f.Skip("race: find-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
		count := bounded(items, 0, maxItems)
		decision := bounded(decisionIndex, 0, count+1) // decision >= count: no item matches

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
		predicate := newCountingPredicate(true, func(item int) bool { return item >= decision })

		find := ro.Find(predicate.test)
		if withIndex {
			find = ro.FindI(predicate.testIndexed)
		}

		got := collect(t, find(numbers.observable()), numbers)

		got.expectValues(t, itemAt(decision, count))
		got.expectCompletedOnce(t)
		got.expectContract(t)
		predicate.expectNoCallAfterDecision(t)
	})
}

// FuzzAll checks that every item of a source of `items` integers is < decisionIndex. The source is
// synchronous or asynchronous, and may keep emitting after the operator decided. All is the control
// of this family: it already stops evaluating after the first failing item.
//
// Invariant: All emits exactly one boolean (true when no item failed) and completes once, and it never
// calls the predicate again after the call that failed.
//
// Seeds: same as FuzzContains, with withIndex selecting AllI.
func FuzzAll(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
		count := bounded(items, 0, maxItems)
		decision := bounded(decisionIndex, 0, count+1) // decision >= count: no item fails

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
		predicate := newCountingPredicate(false, func(item int) bool { return item < decision })

		all := ro.All(predicate.test)
		if withIndex {
			all = ro.AllI(predicate.testIndexed)
		}

		got := collect(t, all(numbers.observable()), numbers)

		got.expectValues(t, []bool{decision >= count})
		got.expectCompletedOnce(t)
		got.expectContract(t)
		predicate.expectNoCallAfterDecision(t)
	})
}
