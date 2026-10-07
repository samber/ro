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
	"sync"
	"testing"
	"time"

	"github.com/samber/ro"
)

const (
	// maxTicks bounds the number of ticks a boundary source emits, to keep one iteration fast.
	maxTicks = 12

	// maxSpinsBeforeUnsubscribe bounds the scheduler yields that delay an external Unsubscribe.
	maxSpinsBeforeUnsubscribe = 24

	// shortestPeriodMicros and longestPeriodMicros bound the period of a timer operator. A period of the
	// same order as the time between two items is what makes a timer fire while an item is in flight.
	shortestPeriodMicros = 100
	longestPeriodMicros  = 2000

	// longestTickPauseMicros bounds the pause between two ticks of an asynchronous boundary source.
	longestTickPauseMicros = 300
)

// timerPeriod maps a fuzz input to the period of a timer operator.
func timerPeriod(micros uint16) time.Duration {
	return time.Duration(bounded(micros, shortestPeriodMicros, longestPeriodMicros)) * time.Microsecond
}

// tickingBoundary emits `ticks` ticks and never completes, like a boundary Observable that outlives
// the source it controls. An asynchronous boundary pauses between ticks, so ticks land between items.
func tickingBoundary(ticks int, async bool, yieldPattern uint8) ro.Observable[int] {
	if !async {
		return newSource(ticks, false).neverEnding().observable()
	}

	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
		stopped := make(chan struct{})

		var stop sync.Once

		go func() {
			for tick := 0; tick < ticks; tick++ {
				if isClosed(stopped) {
					return
				}

				yieldAt(int64(yieldPattern)+maxTicks, tick)
				time.Sleep(time.Duration((int(yieldPattern)*37+tick*101)%longestTickPauseMicros) * time.Microsecond)
				destination.NextWithContext(ctx, tick)
			}
		}()

		return func() { stop.Do(func() { close(stopped) }) }
	})
}

// streamEnd says how a target ends its stream: by waiting for the terminal notification, or by
// unsubscribing from outside after a few scheduler yields.
type streamEnd struct {
	unsubscribeEarly bool
	spins            int
	yieldPattern     uint8
}

func newStreamEnd(unsubscribeEarly bool, spinsBeforeUnsubscribe, yieldPattern uint8) streamEnd {
	return streamEnd{
		unsubscribeEarly: unsubscribeEarly,
		spins:            bounded(spinsBeforeUnsubscribe, 0, maxSpinsBeforeUnsubscribe),
		yieldPattern:     yieldPattern,
	}
}

// waitOrUnsubscribe is called once Subscribe returned: it unsubscribes after the spins, or waits for
// the terminal notification. It reports whether the stream ran to its natural end.
func (end streamEnd) waitOrUnsubscribe(t *testing.T, unsubscribe func(), stream progress) (completed bool) {
	t.Helper()

	if end.unsubscribeEarly {
		for spin := 0; spin < end.spins; spin++ {
			yieldAt(int64(end.yieldPattern), spin)
		}

		unsubscribe()

		return false
	}

	waitUntil(t, "the downstream terminal notification", func() bool { return stream.terminalCount() > 0 })

	return true
}

// observeUntilEnd subscribes to pipeline, ends the stream as end says, then checks that every counted
// upstream was released and that the observer contract held. When the stream ran to its natural end it
// also checks that it completed once without error. completed reports which case it was.
func observeUntilEnd[T any](t *testing.T, pipeline ro.Observable[T], end streamEnd, upstreams ...*subscriptionCounter) (got *recorder[T], completed bool) {
	t.Helper()

	got = newRecorder[T]()

	// Subscribe returns through a channel, not a poll: an early Unsubscribe must follow it
	// immediately, or an asynchronous source finishes before the unsubscription it should race with.
	subscription := subscribeWithinDeadline[T](t, pipeline, got)

	completed = end.waitOrUnsubscribe(t, subscription.Unsubscribe, got)

	for _, upstream := range upstreams {
		upstream.expectAllReleased(t)
	}

	time.Sleep(settleDelay)
	got.expectContract(t)

	if completed {
		got.expectCompletedOnce(t)
	}

	return got, completed
}

// subscribeWithinDeadline subscribes observer to pipeline and fails the test when Subscribe does not
// return, instead of hanging the run.
func subscribeWithinDeadline[T any](t *testing.T, pipeline ro.Observable[T], observer ro.Observer[T]) ro.Subscription {
	t.Helper()

	var subscription ro.Subscription

	runWithinDeadline(t, func() { subscription = pipeline.Subscribe(observer) })

	return subscription
}

// flattened concatenates the batches in order.
func flattened(batches [][]int) []int {
	all := []int{}
	for _, batch := range batches {
		all = append(all, batch...)
	}

	return all
}

// expectCompleteInOrder checks that got is exactly 0..count-1: nothing lost, nothing duplicated, nothing reordered.
func expectCompleteInOrder(t *testing.T, got []int, count int) {
	t.Helper()

	if len(got) != count {
		t.Fatalf("got %d items %v, want %d (items lost or duplicated)", len(got), got, count)
	}

	for index, value := range got {
		if value != index {
			t.Fatalf("item %d is %d: out of order in %v", index, value, got)
		}
	}
}

// expectStrictlyIncreasing checks that got is a subsequence of 0..below-1: every value in range, each
// larger than the previous one. It is what a sampled or throttled stream of 0..below-1 looks like.
func expectStrictlyIncreasing(t *testing.T, got []int, below int) {
	t.Helper()

	for index, value := range got {
		if value < 0 || value >= below {
			t.Fatalf("value %d out of range [0,%d)", value, below)
		}

		if index > 0 && value <= got[index-1] {
			t.Fatalf("values not strictly increasing: %v", got)
		}
	}
}

// runToCompletion subscribes to pipeline, waits for its terminal notification, unsubscribes, then checks
// that every counted upstream was released and that the observer contract held.
func runToCompletion(t *testing.T, pipeline ro.Observable[int], upstreams ...*subscriptionCounter) *recorder[int] {
	t.Helper()

	got := newRecorder[int]()
	subscription := subscribeInBackground(context.Background(), pipeline, got)
	subscription.expectReturn(t)

	waitUntil(t, "a terminal notification", func() bool { return got.terminalCount() > 0 })
	subscription.unsubscribe()

	expectReleasedWithContract(t, got, upstreams)

	return got
}

// runWithTake is runToCompletion with the stream closed from inside the pipeline by ro.Take(limit).
func runWithTake(t *testing.T, pipeline ro.Observable[int], limit int, upstreams ...*subscriptionCounter) *recorder[int] {
	t.Helper()

	return runToCompletion(t, ro.Take[int](int64(limit))(pipeline), upstreams...)
}

// runThenUnsubscribe subscribes to pipeline, unsubscribes from outside once `items` values arrived (or
// the stream ended first), then checks that every counted upstream was released and the contract held.
func runThenUnsubscribe(t *testing.T, pipeline ro.Observable[int], items int, upstreams ...*subscriptionCounter) *recorder[int] {
	t.Helper()

	got := newRecorder[int]()
	subscription := subscribeInBackground(context.Background(), pipeline, got)
	subscription.unsubscribeAfterItems(t, got, items)

	expectReleasedWithContract(t, got, upstreams)

	return got
}

func expectReleasedWithContract(t *testing.T, got *recorder[int], upstreams []*subscriptionCounter) {
	t.Helper()

	for _, upstream := range upstreams {
		upstream.expectAllReleased(t)
	}

	got.expectContract(t)
}
