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
	"context"
	"errors"
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
)

var errPrimitiveTest = errors.New("primitive test failure")

func TestBoundedPivots(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name  string
		value int64
		want  int
	}{
		{"low bound", 0, 3},
		{"inside", 2, 5},
		{"wraps to low bound", 4, 3},
		{"negative", -1, 4},
		{"most negative", -9223372036854775808, 3},
	}

	for _, c := range cases {
		c := c

		t.Run(c.name, func(t *testing.T) {
			t.Parallel()

			// Span is 4 values: [3, 6].
			if got := bounded(c.value, 3, 6); got != c.want {
				t.Fatalf("bounded(%d, 3, 6) = %d, want %d", c.value, got, c.want)
			}
		})
	}
}

func TestSourceVariants(t *testing.T) {
	t.Parallel()

	for _, async := range []bool{false, true} {
		async := async

		t.Run("completes", func(t *testing.T) {
			t.Parallel()

			numbers := newSource(3, async).startingAt(10).yieldingWith(7)
			got := collect(t, numbers.observable(), numbers)

			got.expectValues(t, []int{10, 11, 12})
			got.expectCompletedOnce(t)
			got.expectContract(t)
		})

		t.Run("fails at end", func(t *testing.T) {
			t.Parallel()

			numbers := newSource(2, async).failingAtEnd(errPrimitiveTest)
			got := collect(t, numbers.observable(), numbers)

			got.expectValues(t, []int{0, 1})
			got.expectFailedOnce(t)
		})

		t.Run("never ends", func(t *testing.T) {
			t.Parallel()

			numbers := newSource(2, async).neverEnding()
			got := collect(t, numbers.observable(), numbers)

			got.expectValues(t, []int{0, 1})

			if got.terminalCount() != 0 {
				t.Fatalf("a never-ending source terminated")
			}
		})
	}
}

func TestIgnoringStopEmitsEverything(t *testing.T) {
	t.Parallel()

	for _, async := range []bool{false, true} {
		numbers := newSource(5, async).ignoringStop(true)
		got := collect(t, ro.Take[int](1)(numbers.observable()), numbers)

		got.expectValues(t, []int{0})
		got.expectCompletedOnce(t)
	}
}

func TestCountSubscriptions(t *testing.T) {
	t.Parallel()

	counter := &subscriptionCounter{}
	numbers := newSource(3, false).neverEnding()

	subscription := countSubscriptions(counter, numbers.observable()).Subscribe(newRecorder[int]())
	if counter.activeCount() != 1 || counter.totalCount() != 1 {
		t.Fatalf("active=%d total=%d, want 1 and 1", counter.activeCount(), counter.totalCount())
	}

	subscription.Unsubscribe()
	counter.expectAllReleased(t)

	if counter.totalCount() != 1 {
		t.Fatalf("total=%d, want 1", counter.totalCount())
	}
}

func TestOverlapGuardCountsNestedCalls(t *testing.T) {
	t.Parallel()

	guard := &overlapGuard{}

	guard.enter()
	guard.enter()
	guard.leave()
	guard.leave()

	if guard.overlapCount() != 1 {
		t.Fatalf("overlapCount=%d, want 1", guard.overlapCount())
	}
}

func TestCountingPredicateCountsCallsAfterDecision(t *testing.T) {
	t.Parallel()

	predicate := newCountingPredicate(true, func(item int) bool { return item >= 2 })

	predicate.test(0)
	predicate.test(2) // the decision
	predicate.expectNoCallAfterDecision(t)

	predicate.testIndexed(3, 3)

	if got := atomic.LoadInt32(&predicate.callsAfterDecision); got != 1 {
		t.Fatalf("callsAfterDecision=%d, want 1", got)
	}
}

func TestUnsubscribeAfterItems(t *testing.T) {
	t.Parallel()

	numbers := newSource(5, true).neverEnding()
	got := newRecorder[int]()

	background := subscribeInBackground(context.Background(), numbers.observable(), got)
	background.expectReturn(t)
	background.unsubscribeAfterItems(t, got, 3)

	runWithinDeadline(t, numbers.waitForProducers)
}

func TestCancelAfterItems(t *testing.T) {
	t.Parallel()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	numbers := newSource(5, false).neverEnding()
	got := newRecorder[int]()

	background := subscribeInBackground(ctx, numbers.observable(), got)
	background.expectReturn(t)

	cancelAfterItems(t, cancel, got, 5)

	if ctx.Err() == nil {
		t.Fatal("context not canceled")
	}

	background.unsubscribe()
}
