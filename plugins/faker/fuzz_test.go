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

package rofaker

import (
	"context"
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

const (
	// fuzzMaxItems keeps one scenario fast while still spanning several channel hand-offs.
	fuzzMaxItems = 64

	// fuzzInfiniteCap stops a generator that was never told to stop, so a bug cannot spin forever.
	fuzzInfiniteCap = 2_000_000

	// maskAsync puts a goroutine hop between the source and the observer.
	maskAsync = 1 << 0
)

// FuzzGenerateInfiniteTake checks that an infinite Generate stops calling its generator once
// Take is satisfied. The async variant puts a goroutine hop (ObserveOn) after Take.
func FuzzGenerateInfiniteTake(f *testing.F) {
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		take := 1 + int64(mask>>1)%fuzzMaxItems
		async := mask&maskAsync != 0

		var calls int64
		var received int64
		completed := make(chan struct{})

		obs := ro.Take[int](take)(Generate(func() (int, error) {
			if n := atomic.AddInt64(&calls, 1); n < fuzzInfiniteCap {
				return int(n), nil
			}
			return 0, context.DeadlineExceeded
		}))
		if async {
			obs = ro.ObserveOn[int](1)(obs)
		}

		sub := obs.Subscribe(ro.NewObserver(
			func(int) { atomic.AddInt64(&received, 1) },
			func(error) { close(completed) },
			func() { close(completed) },
		))
		defer sub.Unsubscribe()

		xfuzz.WaitChan(t, completed, "Take completion")

		if got := atomic.LoadInt64(&calls); got >= fuzzInfiniteCap {
			t.Fatalf("seed=%d: generator ran to its cap (%d calls) for Take(%d)", seed, got, take)
		}
		if got := atomic.LoadInt64(&received); got != take {
			t.Fatalf("seed=%d: received %d values, want %d", seed, got, take)
		}
	})
}

// FuzzGenerateNCancel checks that cancelling the subscriber context mid-stream ends GenerateN
// with the context error and never emits after it.
func FuzzGenerateNCancel(f *testing.F) {
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		cancelAt := 1 + int64(mask>>1)%fuzzMaxItems

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		var calls int64
		values, _, err := ro.CollectWithContext(ctx, GenerateN(fuzzMaxItems*2, func() (int64, error) {
			n := atomic.AddInt64(&calls, 1)
			if n == cancelAt {
				cancel()
			}
			return n, nil
		}))

		if err != context.Canceled {
			t.Fatalf("seed=%d: err = %v, want context.Canceled", seed, err)
		}
		if int64(len(values)) != cancelAt {
			t.Fatalf("seed=%d: received %d values, want %d", seed, len(values), cancelAt)
		}
	})
}
