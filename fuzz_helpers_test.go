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

package ro

import (
	"context"
	"runtime"
	"sync/atomic"
	"testing"
	"time"
)

// Helpers shared by the Fuzz* targets of the root package. They use sync/atomic
// functions instead of atomic.Int32 because go.mod declares go 1.18.

const (
	// fuzzDeadline bounds every blocking wait of a fuzz iteration. A hang is a bug, so the
	// bound is generous enough for a loaded CI machine but short enough to fail fast.
	fuzzDeadline = 5 * time.Second

	// fuzzMaxItems bounds the number of items a fuzz input can request, to keep one iteration fast.
	fuzzMaxItems = 64

	// fuzzMaxGoroutines bounds the number of concurrent actors a fuzz input can request.
	fuzzMaxGoroutines = 8
)

// fuzzBound maps any fuzz input to [lo, hi], so that negative or huge values stay valid scenarios.
func fuzzBound(v int64, lo, hi int) int {
	span := int64(hi - lo + 1)

	r := v % span
	if r < 0 {
		r = -r
	}

	return lo + int(r)
}

// fuzzJitter yields the processor at positions chosen by seed, so that different seeds
// explore different interleavings. step identifies the call site inside one iteration.
func fuzzJitter(seed int64, step int) {
	// Mix seed and step so neighbouring steps do not yield together.
	mixed := uint64(seed)*6364136223846793005 + uint64(step)*1442695040888963407 //nolint:gosec // wrap-around is intended.

	switch mixed >> 61 { // 3 top bits: 0..7.
	case 0, 1:
		runtime.Gosched()
	case 2:
		time.Sleep(time.Microsecond)
	}
}

// serialGuard detects overlapping calls: wrap the body of an observer callback with
// enter/leave and check overlapped() at the end of the test.
type serialGuard struct {
	inside     int32
	overlapCnt int32
}

func (g *serialGuard) enter() {
	if atomic.AddInt32(&g.inside, 1) > 1 {
		atomic.AddInt32(&g.overlapCnt, 1)
	}
}

func (g *serialGuard) leave() { atomic.AddInt32(&g.inside, -1) }

func (g *serialGuard) overlapped() int { return int(atomic.LoadInt32(&g.overlapCnt)) }

// activeCounter counts live subscriptions to a source. Use it with Defer + TapOnSubscribe-like
// bookkeeping to assert that an operator unsubscribed from every upstream.
type activeCounter struct {
	active int32
	total  int32
}

func (c *activeCounter) open() {
	atomic.AddInt32(&c.active, 1)
	atomic.AddInt32(&c.total, 1)
}

func (c *activeCounter) close() { atomic.AddInt32(&c.active, -1) }

func (c *activeCounter) activeCount() int { return int(atomic.LoadInt32(&c.active)) }

func (c *activeCounter) totalCount() int { return int(atomic.LoadInt32(&c.total)) }

// track wraps source so that every subscription increments the counter until it is torn down.
func trackSubscriptions[T any](c *activeCounter, source Observable[T]) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		c.open()

		sub := source.SubscribeWithContext(ctx, destination)

		return func() {
			sub.Unsubscribe()
			c.close()
		}
	})
}

// fuzzSource emits 0..n-1 then completes. Every fuzz target must run against BOTH kinds:
// a synchronous source emits inside Subscribe (teardown is registered only after it returns),
// while an asynchronous source emits from its own goroutine (Next races Unsubscribe and Complete).
// Derive async from a bit of the fuzz input so the corpus covers both.
func fuzzSource(seed int64, n int, async bool) Observable[int] {
	if !async {
		return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			for i := 0; i < n && !destination.IsClosed(); i++ {
				destination.NextWithContext(ctx, i)
			}

			destination.CompleteWithContext(ctx)

			return nil
		})
	}

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		done := make(chan struct{})

		go func() {
			for i := 0; i < n; i++ {
				select {
				case <-done:
					return
				default:
				}

				fuzzJitter(seed, i)
				destination.NextWithContext(ctx, i)
			}

			destination.CompleteWithContext(ctx)
		}()

		return func() { close(done) }
	})
}

// fuzzIsAsync reads the sync/async choice of source number idx from a bitmask fuzz input.
func fuzzIsAsync(mask uint8, idx int) bool { return mask&(1<<uint(idx%8)) != 0 }

// fuzzWaitFor polls cond until it is true or fuzzDeadline expires.
func fuzzWaitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()

	deadline := time.Now().Add(fuzzDeadline)
	for !cond() {
		if time.Now().After(deadline) {
			t.Fatalf("timeout waiting for: %s", what)
		}

		time.Sleep(time.Millisecond)
	}
}
