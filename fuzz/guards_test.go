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
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
)

// overlapGuard detects overlapping calls: wrap the body of a callback with enter and leave,
// then call expectNone at the end of the test.
type overlapGuard struct {
	inside   int32
	overlaps int32
}

func (g *overlapGuard) enter() {
	if atomic.AddInt32(&g.inside, 1) > 1 {
		atomic.AddInt32(&g.overlaps, 1)
	}
}

func (g *overlapGuard) leave() { atomic.AddInt32(&g.inside, -1) }

func (g *overlapGuard) overlapCount() int { return int(atomic.LoadInt32(&g.overlaps)) }

func (g *overlapGuard) expectNone(t *testing.T) {
	t.Helper()

	if count := g.overlapCount(); count != 0 {
		t.Fatalf("%d overlapping calls", count)
	}
}

// subscriptionCounter counts the live subscriptions to the sources wrapped by countSubscriptions.
type subscriptionCounter struct {
	active int32
	total  int32
}

func (c *subscriptionCounter) activeCount() int { return int(atomic.LoadInt32(&c.active)) }

func (c *subscriptionCounter) totalCount() int { return int(atomic.LoadInt32(&c.total)) }

// expectAllReleased waits for every counted subscription to be torn down. The teardown of an
// asynchronous pipeline may run after the terminal notification, hence the wait.
func (c *subscriptionCounter) expectAllReleased(t *testing.T) {
	t.Helper()

	waitUntil(t, "every upstream subscription to be released", func() bool { return c.activeCount() == 0 })
}

// countSubscriptions wraps source so that each subscription is counted from Subscribe until its teardown.
func countSubscriptions[T any](counter *subscriptionCounter, source ro.Observable[T]) ro.Observable[T] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[T]) ro.Teardown {
		atomic.AddInt32(&counter.active, 1)
		atomic.AddInt32(&counter.total, 1)

		subscription := source.SubscribeWithContext(ctx, destination)

		return func() {
			subscription.Unsubscribe()
			atomic.AddInt32(&counter.active, -1)
		}
	})
}
