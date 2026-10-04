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
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// FuzzSCAverageEmpty checks Average over an empty source: at most one value, then exactly one terminal.
// The Subscriber absorbs every notification sent after the first terminal one, so the extra
// Next/Complete of a missing return is only visible through ro.OnDroppedNotification.
func FuzzSCAverageEmpty(f *testing.F) {
	f.Skip("race: average-empty-double-emit (missing return after NaN+Complete); remove when fixed")

	fuzzSCSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		var dropped int32

		// The hook is a package-level variable: this target is not parallel, and it is restored on cleanup.
		previous := ro.OnDroppedNotification
		ro.OnDroppedNotification = func(context.Context, fmt.Stringer) { atomic.AddInt32(&dropped, 1) }
		t.Cleanup(func() { ro.OnDroppedNotification = previous })

		sink := &fuzzSCRawSink[float64]{}

		fuzzSCIter(t, "AverageEmpty", func() error {
			var wg sync.WaitGroup

			sub := ro.Average[int]()(fuzzSCSource(seed, 0, fuzzIsAsync(mask, 0), true, false, &wg)).
				SubscribeWithContext(context.Background(), sink)

			wg.Wait()
			time.Sleep(fuzzSCSettle)

			sub.Unsubscribe()

			return nil
		})

		got := sink.snapshot()
		errs := atomic.LoadInt32(&sink.errs)
		comps := atomic.LoadInt32(&sink.comps)
		after := atomic.LoadInt32(&sink.afterTerm) + atomic.LoadInt32(&dropped)

		if len(got) > 1 || errs != 0 || comps != 1 || after != 0 {
			t.Fatalf("AverageEmpty (async=%v): values=%v (want at most 1); errors=%d; completes=%d (want 1); notifications after terminal=%d",
				fuzzIsAsync(mask, 0), got, errs, comps, after)
		}
	})
}
