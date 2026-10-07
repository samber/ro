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
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
)

const (
	// maxLoopTake bounds how many items a loop target takes, so that a loop which ignores Take is slow
	// rather than endless.
	maxLoopTake = 12

	// maxLoopItemsPerRun and maxFiniteLoopRuns bound the size of one run of a looping source and the
	// number of runs of a finite loop.
	maxLoopItemsPerRun = 4
	maxFiniteLoopRuns  = 3

	// unlimitedLoop is a loop bound that only a downstream stop can reach.
	unlimitedLoop = 1 << 30
)

// loopCondition is the condition of ro.While and ro.DoWhile: true for the first `limit` evaluations.
// The test marks it abandoned when it ends, so a buggy loop that ignores a closed downstream spins
// only until the test fails, instead of for ever.
type loopCondition struct {
	evaluations int32
	limit       int32
	abandoned   int32
}

func newLoopCondition(t *testing.T, limit int) *loopCondition {
	t.Helper()

	condition := &loopCondition{limit: int32(limit)} //nolint:gosec // callers pass a small bound or unlimitedLoop.

	t.Cleanup(func() { atomic.StoreInt32(&condition.abandoned, 1) })

	return condition
}

func (c *loopCondition) holds() bool {
	return atomic.LoadInt32(&c.abandoned) == 0 && atomic.AddInt32(&c.evaluations, 1) <= c.limit
}

// subscriptionsToDeliver is how many runs of a source of itemsPerRun items are needed to deliver
// `items` items: a loop must not open another run afterwards.
func subscriptionsToDeliver(items, itemsPerRun int) int {
	return (items + itemsPerRun - 1) / itemsPerRun
}

// expectLoopStoppedByTake checks a loop that Take(takeCount) closed: exactly takeCount items were
// delivered and no run was opened after the downstream closed.
func expectLoopStoppedByTake(t *testing.T, got *recorder[int], upstream *subscriptionCounter, takeCount, itemsPerRun int) {
	t.Helper()

	if delivered := got.valueCount(); delivered != takeCount {
		t.Fatalf("take(%d) delivered %d values", takeCount, delivered)
	}

	if runs, enough := upstream.totalCount(), subscriptionsToDeliver(takeCount, itemsPerRun); runs > enough {
		t.Fatalf("%d source subscriptions after the downstream closed, %d were enough", runs, enough)
	}
}

// expectFiniteLoop checks a loop that ran `runs` times to its end: every run delivered its items and
// the stream completed once.
func expectFiniteLoop(t *testing.T, got *recorder[int], upstream *subscriptionCounter, runs, itemsPerRun int) {
	t.Helper()

	if subscriptions, delivered := upstream.totalCount(), got.valueCount(); subscriptions != runs || delivered != runs*itemsPerRun {
		t.Fatalf("finite loop: %d subscriptions (want %d), %d values (want %d)", subscriptions, runs, delivered, runs*itemsPerRun)
	}

	got.expectCompletedOnce(t)
}

// runLoopStoppedByTake runs `loop` over a source of itemsPerRun items, closes the stream with Take(takeCount),
// and returns the recorder and the counter of source subscriptions. When the source fails at the end of each
// run, the loop is a retry.
func runLoopStoppedByTake(t *testing.T, loop func(ro.Observable[int]) ro.Observable[int], source *source, takeCount int) (*recorder[int], *subscriptionCounter) {
	t.Helper()

	upstream := &subscriptionCounter{}
	got := runWithTake(t, loop(countSubscriptions(upstream, source.observable())), takeCount, upstream)

	return got, upstream
}

// runFiniteLoop runs `loop` over a source of itemsPerRun items until the loop ends on its own.
func runFiniteLoop(t *testing.T, loop func(ro.Observable[int]) ro.Observable[int], source *source) (*recorder[int], *subscriptionCounter) {
	t.Helper()

	upstream := &subscriptionCounter{}
	got := runToCompletion(t, loop(countSubscriptions(upstream, source.observable())), upstream)

	return got, upstream
}

// expectValueCount checks how many values were delivered, whatever their order.
func expectValueCount(t *testing.T, got *recorder[int], want int) {
	t.Helper()

	if delivered := got.valueCount(); delivered != want {
		t.Fatalf("delivered %d values, want %d", delivered, want)
	}
}
