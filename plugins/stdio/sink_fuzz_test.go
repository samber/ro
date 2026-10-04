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

package rostdio

import (
	"context"
	"math/rand"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

// FuzzIOWriterStopsAfterWriteError checks that the sink stops writing once it reported a write error.
// The source is either a synchronous loop or a goroutine, neither of which watches the sink.
func FuzzIOWriterStopsAfterWriteError(f *testing.F) {
	f.Skip("race: stdio-writer-keeps-writing-after-error; remove when fixed")
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		count := 3 + int(mask>>1)%fuzzMaxItems
		writer := &xfuzz.FailingWriter{FailAt: int(seed&0x7fffffff) % (count - 1)}
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving

		emit := func(ctx context.Context, dst ro.Observer[[]byte]) {
			for i := 0; i < count; i++ {
				if async && rng.Intn(3) == 0 {
					runtime.Gosched()
				}
				dst.NextWithContext(ctx, []byte("payload"))
			}
			dst.CompleteWithContext(ctx)
		}
		src := ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[[]byte]) ro.Teardown {
			if async {
				go emit(ctx, dst)
			} else {
				emit(ctx, dst)
			}
			return nil
		})

		var errSeen int32
		sub := NewIOWriter(writer)(src).Subscribe(ro.NewObserver(
			func(int) {},
			func(error) { atomic.StoreInt32(&errSeen, 1) },
			func() {},
		))
		defer sub.Unsubscribe()

		// Writes happen on the source's goroutine for async sources: wait until all `count` items went through
		// the writer, or, if the sink stopped writing, until things stayed quiet for a short window.
		const quiet = 30 * time.Millisecond
		deadline := time.Now().Add(xfuzz.Deadline)
		lastCalls, lastChange := int64(-1), time.Now()
		for writer.Calls() < int64(count) && time.Now().Before(deadline) {
			if c := writer.Calls(); c != lastCalls {
				lastCalls, lastChange = c, time.Now()
			}
			if atomic.LoadInt32(&errSeen) == 1 && time.Since(lastChange) > quiet {
				break
			}
			time.Sleep(time.Millisecond)
		}

		if atomic.LoadInt32(&errSeen) == 0 {
			t.Fatalf("writer error never reached the observer (count=%d failAt=%d)", count, writer.FailAt)
		}
		if n := writer.CallsAfterFail(); n > 0 {
			t.Fatalf("async=%v: %d writes were attempted after the first write error (failAt=%d)", async, n, writer.FailAt)
		}
	})
}
