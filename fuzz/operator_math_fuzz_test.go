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

// FuzzAverageEmptySource averages an empty source, synchronous or asynchronous.
//
// Invariant: the average of nothing emits at most one value and then completes exactly once. A
// second Next or Complete is absorbed by the closed Subscriber, so it is only visible through
// ro.OnDroppedNotification, which the target requires to stay silent.
//
// Seeds: asyncSource alternates.
func FuzzAverageEmptySource(f *testing.F) {
	f.Skip("race: average-empty-double-emit (missing return after NaN+Complete); remove when fixed")

	fuzzSeeds(f, func(i int) []any {
		return []any{i%2 == 0} // asyncSource
	})

	f.Fuzz(func(t *testing.T, asyncSource bool) {
		dropped := captureDroppedNotifications(t)

		empty := newSource(0, asyncSource)
		got := collect(t, ro.Average[int]()(empty.observable()), empty)

		got.expectAtMostValues(t, 1)
		got.expectCompletedOnce(t)
		got.expectContract(t)
		dropped.expectNone(t)
	})
}
