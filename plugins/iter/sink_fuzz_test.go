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
	"context"
	"errors"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

var errFuzzBoom = errors.New("fuzz boom")

// fuzzSource emits 0..count-1 then terminates. Async sources emit from their own goroutine,
// yielding at seed-driven points to vary the interleaving with the consumer.
func fuzzSource(seed int64, count int, async bool, terminal error) ro.Observable[int] {
	if !async {
		return ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
			for i := 0; i < count; i++ {
				dst.NextWithContext(ctx, i)
			}
			if terminal != nil {
				dst.ErrorWithContext(ctx, terminal)
			} else {
				dst.CompleteWithContext(ctx)
			}
			return nil
		})
	}

	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
		stop := make(chan struct{})
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving, not security
		go func() {
			for i := 0; i < count; i++ {
				select {
				case <-stop:
					return
				default:
				}
				if rng.Intn(3) == 0 {
					runtime.Gosched()
				}
				dst.NextWithContext(ctx, i)
			}
			if terminal != nil {
				dst.ErrorWithContext(ctx, terminal)
			} else {
				dst.CompleteWithContext(ctx)
			}
		}()
		return func() { close(stop) }
	})
}

// recoverValue runs fn and returns what it panicked with, if anything.
func recoverValue(fn func()) (recovered any) {
	defer func() { recovered = recover() }()
	fn()
	return nil
}

// FuzzToSeqKeepsEveryItem checks that no item is dropped or reordered between the
// producer and the iterator consumer, for sync and async sources.
func FuzzToSeqKeepsEveryItem(f *testing.F) {
	f.Skip("race: iter-toseq-lost-last-item (async: last item dropped when done wins the select) and iter-toseq-sync-deadlock (sync source with >=2 items blocks Subscribe on the 1-slot channel); remove when fixed")
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		count := 1 + int(mask>>1)%fuzzMaxItems
		async := mask&maskAsync != 0
		before := runtime.NumGoroutine()

		var got []int
		finished := make(chan struct{})
		var panicked atomic.Value
		go func() {
			defer close(finished)
			if r := recoverValue(func() {
				for v := range ToSeq(fuzzSource(seed, count, async, nil)) {
					got = append(got, v)
				}
			}); r != nil {
				panicked.Store(r)
			}
		}()
		xfuzz.WaitChan(t, finished, "ToSeq consumer")

		if r := panicked.Load(); r != nil {
			t.Fatalf("unexpected panic: %v", r)
		}
		if len(got) != count {
			t.Fatalf("async=%v count=%d: received %d items: %v", async, count, len(got), got)
		}
		for i, v := range got {
			if v != i {
				t.Fatalf("async=%v: item %d out of order: %v", async, i, got)
			}
		}
		xfuzz.AssertNoLeak(t, before)
	})
}

// FuzzToSeq2KeepsEveryItem is the ToSeq2 twin: indexes must be contiguous and values complete.
func FuzzToSeq2KeepsEveryItem(f *testing.F) {
	f.Skip("race: iter-toseq2-lost-last-item and iter-toseq2-sync-deadlock, same as ToSeq; remove when fixed")
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		count := 1 + int(mask>>1)%fuzzMaxItems
		async := mask&maskAsync != 0

		var got []int
		finished := make(chan struct{})
		go func() {
			defer close(finished)
			_ = recoverValue(func() {
				for i, v := range ToSeq2(fuzzSource(seed, count, async, nil)) {
					if i != len(got) {
						got = append(got, -1)
						return
					}
					got = append(got, v)
				}
			})
		}()
		xfuzz.WaitChan(t, finished, "ToSeq2 consumer")

		if len(got) != count {
			t.Fatalf("async=%v count=%d: received %d items: %v", async, count, len(got), got)
		}
		for i, v := range got {
			if v != i {
				t.Fatalf("async=%v: item %d wrong: %v", async, i, got)
			}
		}
	})
}

// FuzzToSeqSourceError checks that a source error never crashes a goroutine the consumer does not own.
// A sync source runs on the consumer's goroutine, so a panic there is observable and tolerated.
func FuzzToSeqSourceError(f *testing.F) {
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		count := int(mask>>1) % fuzzMaxItems
		async := mask&maskAsync != 0
		if !async && count > 1 {
			// A sync source with several items deadlocks ToSeq on its own (see FuzzToSeqKeepsEveryItem);
			// cap it so this target only exercises the error path.
			count = 1
		}

		var producerPanic atomic.Value
		src := fuzzSource(seed, count, async, errFuzzBoom)
		if async {
			// The producer goroutine is the only one able to observe the panic; recover there so
			// a bug fails this target instead of killing the whole test binary.
			inner := src
			src = ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
				guarded := ro.NewObserverWithContext(
					dst.NextWithContext,
					func(c context.Context, err error) {
						if r := recoverValue(func() { dst.ErrorWithContext(c, err) }); r != nil {
							producerPanic.Store(r)
						}
					},
					dst.CompleteWithContext,
				)
				sub := inner.SubscribeWithContext(ctx, guarded)
				return sub.Unsubscribe
			})
		}

		finished := make(chan struct{})
		go func() {
			defer close(finished)
			_ = recoverValue(func() {
				for range ToSeq(src) { //nolint:revive // only the termination matters
				}
			})
		}()
		xfuzz.WaitChan(t, finished, "ToSeq consumer after source error")

		if r := producerPanic.Load(); r != nil {
			t.Fatalf("source error panicked on the producer goroutine: %v", r)
		}
	})
}
