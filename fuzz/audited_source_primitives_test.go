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
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// readPause maps a fuzz input to the time a slow downstream takes to read one notification.
func readPause(micros uint8) time.Duration { return time.Duration(micros) * time.Microsecond }

// pauseEveryNthItem is how often an auditedSource with a pause sleeps: before items 0, 3, 6...
const pauseEveryNthItem = 3

// auditedSource emits 0..items-1 then completes, synchronously or from its own goroutine. It never
// looks at unsubscription, so it keeps emitting into an operator that already stopped. It also keeps
// what a plain source cannot:
//   - upstream counts the live subscriptions to it;
//   - a panic raised by a downstream call is counted, instead of being lost in an operator-internal recover;
//   - the goroutines of an asynchronous source are tracked, to tell whether they were released.
type auditedSource struct {
	items int
	async bool
	// pause makes the source sleep before every pauseEveryNthItem-th item, so that timers of the
	// operator under test fire between two items.
	pause time.Duration

	upstream subscriptionCounter

	producers sync.WaitGroup
	panics    int32
	panicText atomic.Value
}

func newAuditedSource(items int, async bool, pause time.Duration) *auditedSource {
	return &auditedSource{items: items, async: async, pause: pause}
}

func (s *auditedSource) observable() ro.Observable[int] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
		atomic.AddInt32(&s.upstream.active, 1)
		atomic.AddInt32(&s.upstream.total, 1)

		if s.async {
			s.producers.Add(1)

			go func() {
				defer s.producers.Done()

				s.emit(ctx, destination)
			}()
		} else {
			s.emit(ctx, destination)
		}

		return func() { atomic.AddInt32(&s.upstream.active, -1) }
	})
}

func (s *auditedSource) emit(ctx context.Context, destination ro.Observer[int]) {
	defer func() {
		if recovered := recover(); recovered != nil {
			atomic.AddInt32(&s.panics, 1)
			s.panicText.Store(toText(recovered))
		}
	}()

	for item := 0; item < s.items; item++ {
		if s.pause > 0 && item%pauseEveryNthItem == 0 {
			time.Sleep(s.pause)
		}

		// Yields vary with the number of items, so that different inputs interleave differently.
		yieldAt(int64(s.items), item)
		destination.NextWithContext(ctx, item)
	}

	destination.CompleteWithContext(ctx)
}

func toText(recovered any) string {
	if err, isError := recovered.(error); isError {
		return err.Error()
	}

	if text, isText := recovered.(string); isText {
		return text
	}

	return "non-text panic value"
}

// expectCleanEnd checks that the producer goroutines were released, that no downstream call panicked,
// and that every subscription to the source was torn down.
func (s *auditedSource) expectCleanEnd(t *testing.T) {
	t.Helper()

	expectWaitGroupDone(t, "producer goroutine (blocked after the downstream stopped)", &s.producers)

	if count := atomic.LoadInt32(&s.panics); count > 0 {
		text, _ := s.panicText.Load().(string)
		t.Fatalf("%d panic(s) while emitting: %s", count, text)
	}

	s.upstream.expectAllReleased(t)
}
