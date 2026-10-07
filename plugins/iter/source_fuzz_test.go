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

package roiter

import (
	"iter"
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

// FuzzFromSeqInfiniteTake checks that an infinite iter.Seq stops once downstream is satisfied.
// The async variant puts a goroutine hop (ObserveOn) between FromSeq and the observer.
func FuzzFromSeqInfiniteTake(f *testing.F) {
	f.Skip("race: iter-fromseq-ignores-downstream-close (infinite Seq never stopped after Take); remove when fixed")
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		take := 1 + int64(mask>>1)%fuzzMaxItems
		async := mask&maskAsync != 0

		var yielded atomic.Int64
		loopDone := make(chan struct{})
		var infinite iter.Seq[int] = func(yield func(int) bool) {
			defer close(loopDone)
			for i := 0; i < fuzzInfiniteCap; i++ {
				yielded.Add(1)
				if !yield(i) {
					return
				}
			}
		}

		var received atomic.Int64
		completed := make(chan struct{})
		obs := ro.Take[int](take)(FromSeq(infinite))
		if async {
			obs = ro.ObserveOn[int](1)(obs)
		}

		// FromSeq blocks Subscribe while it iterates, hence the goroutine.
		go func() {
			sub := obs.Subscribe(ro.NewObserver(
				func(int) { received.Add(1) },
				func(error) {},
				func() { close(completed) },
			))
			defer sub.Unsubscribe()
		}()

		xfuzz.WaitChan(t, completed, "Take completion")
		xfuzz.WaitChan(t, loopDone, "infinite iterator stop after Take")

		if got := yielded.Load(); got >= fuzzInfiniteCap {
			t.Fatalf("seed=%d: iterator ran to its cap (%d values) for Take(%d)", seed, got, take)
		}
		if received.Load() != take {
			t.Fatalf("received %d, want %d", received.Load(), take)
		}
	})
}
