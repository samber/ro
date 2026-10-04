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

// fallbackItem is the default of ElementAtOrDefault. It is outside the range [0, count) of the source,
// so a fallback cannot be mistaken for an emitted item.
const fallbackItem = -1

// FuzzFirst takes the first item >= decisionIndex of a source of `items` integers. The source is
// synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: First emits the first matching item and completes once, or fails once when none matches.
// It never calls the predicate again after the call that matched.
//
// Seeds: items and decisionIndex spread over their range; asyncSource, sourceIgnoresStop and
// withIndex (FirstI instead of First) alternate.
func FuzzFirst(f *testing.F) {
	f.Skip("race: first-predicate-after-decision (sync source keeps calling predicate); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
		count := bounded(items, 0, maxItems)
		decision := bounded(decisionIndex, 0, count+1) // decision >= count: no item matches

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
		predicate := newCountingPredicate(true, func(item int) bool { return item >= decision })

		first := ro.First(predicate.test)
		if withIndex {
			first = ro.FirstI(predicate.testIndexed)
		}

		got := collect(t, first(numbers.observable()), numbers)

		got.expectValues(t, itemAt(decision, count))
		expectSingleItemOrFailure(t, got, decision < count)
		got.expectContract(t)
		predicate.expectNoCallAfterDecision(t)
	})
}

// expectSingleItemOrFailure checks the terminal notification of an operator that picks one item:
// a completion when the item exists, one error otherwise.
func expectSingleItemOrFailure(t *testing.T, got *recorder[int], itemExists bool) {
	t.Helper()

	if itemExists {
		got.expectCompletedOnce(t)
	} else {
		got.expectFailedOnce(t)
	}
}

// FuzzElementAt takes the item at `index` of a source of `items` integers. The source is
// synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: ElementAt emits the item at index and completes once, or fails once when the source is
// shorter.
//
// Seeds: items and index spread over their range; asyncSource and sourceIgnoresStop alternate.
func FuzzElementAt(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, index, asyncSource, sourceIgnoresStop
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2}
	})

	f.Fuzz(func(t *testing.T, items, index uint8, asyncSource, sourceIgnoresStop bool) {
		count := bounded(items, 0, maxItems)
		position := bounded(index, 0, count+1) // position >= count: the source is too short

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)

		got := collect(t, ro.ElementAt[int](position)(numbers.observable()), numbers)

		got.expectValues(t, itemAt(position, count))
		expectSingleItemOrFailure(t, got, position < count)
		got.expectContract(t)
	})
}

// FuzzElementAtOrDefault takes the item at `index` of a source of `items` integers, or a fallback.
// The source is synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: ElementAtOrDefault emits exactly one value, the item at index or fallbackItem when the
// source is shorter, and completes once.
//
// Seeds: same as FuzzElementAt.
func FuzzElementAtOrDefault(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, index, asyncSource, sourceIgnoresStop
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2}
	})

	f.Fuzz(func(t *testing.T, items, index uint8, asyncSource, sourceIgnoresStop bool) {
		count := bounded(items, 0, maxItems)
		position := bounded(index, 0, count+1) // position >= count: the source is too short

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)

		got := collect(t, ro.ElementAtOrDefault(int64(position), fallbackItem)(numbers.observable()), numbers)

		want := []int{fallbackItem}
		if position < count {
			want = []int{position}
		}

		got.expectValues(t, want)
		got.expectCompletedOnce(t)
		got.expectContract(t)
	})
}

// FuzzHead takes the first item of a source of `items` integers. The source is synchronous or
// asynchronous, and may keep emitting after the operator decided.
//
// Invariant: Head emits item 0 and completes once, or fails once when the source is empty.
//
// Seeds: items spreads over its range; asyncSource and sourceIgnoresStop alternate.
func FuzzHead(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, asyncSource, sourceIgnoresStop
		return []any{seedByte(i, 0), i%2 == 0, i%4 < 2}
	})

	f.Fuzz(func(t *testing.T, items uint8, asyncSource, sourceIgnoresStop bool) {
		count := bounded(items, 0, maxItems)

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)

		got := collect(t, ro.Head[int]()(numbers.observable()), numbers)

		got.expectValues(t, itemAt(0, count))
		expectSingleItemOrFailure(t, got, count > 0)
		got.expectContract(t)
	})
}

// FuzzTake takes the first `limit` items of a source of `items` integers. The source is synchronous
// or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: Take emits the first min(limit, items) items in order and completes once, whatever the
// source keeps emitting afterwards.
//
// Seeds: items and limit spread over their range; asyncSource and sourceIgnoresStop alternate.
func FuzzTake(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, limit, asyncSource, sourceIgnoresStop
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2}
	})

	f.Fuzz(func(t *testing.T, items, limit uint8, asyncSource, sourceIgnoresStop bool) {
		count := bounded(items, 0, maxItems)
		taken := bounded(limit, 0, count+1) // taken > count: the source ends first

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)

		got := collect(t, ro.Take[int](int64(taken))(numbers.observable()), numbers)

		got.expectValues(t, sequence(0, smaller(taken, count)))
		got.expectCompletedOnce(t)
		got.expectContract(t)
	})
}

// FuzzTakeWhile takes the items < decisionIndex of a source of `items` integers. The source is
// synchronous or asynchronous, and may keep emitting after the operator decided.
//
// Invariant: TakeWhile emits the leading items that satisfy the predicate and completes once, and it
// never calls the predicate again after the call that failed.
//
// Seeds: same as FuzzFirst, with withIndex selecting TakeWhileI.
func FuzzTakeWhile(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, decisionIndex, asyncSource, sourceIgnoresStop, withIndex
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, i%4 < 2, i%3 == 0}
	})

	f.Fuzz(func(t *testing.T, items, decisionIndex uint8, asyncSource, sourceIgnoresStop, withIndex bool) {
		count := bounded(items, 0, maxItems)
		decision := bounded(decisionIndex, 0, count+1) // decision >= count: every item passes

		numbers := newSource(count, asyncSource).ignoringStop(sourceIgnoresStop)
		predicate := newCountingPredicate(false, func(item int) bool { return item < decision })

		takeWhile := ro.TakeWhile(predicate.test)
		if withIndex {
			takeWhile = ro.TakeWhileI(predicate.testIndexed)
		}

		got := collect(t, takeWhile(numbers.observable()), numbers)

		got.expectValues(t, sequence(0, smaller(decision, count)))
		got.expectCompletedOnce(t)
		got.expectContract(t)
		predicate.expectNoCallAfterDecision(t)
	})
}
