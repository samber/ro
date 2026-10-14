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

package rocsv

import (
	"context"
	"encoding/csv"
	"fmt"
	"math/rand"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
)

// FuzzCSVWriterReportsWriteErrorOnce checks the notification contract of the sink when the underlying writer fails:
// the written count then exactly one error, and nothing after, for sync and async sources.
//
// csv.Writer's bufio layer keeps a sticky error, so "writes attempted after the error" is not observable from here.
func FuzzCSVWriterReportsWriteErrorOnce(f *testing.F) {
	xfuzz.StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		rows := 3 + int(mask>>1)%fuzzMaxItems
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving
		big := strings.Repeat("x", bigFieldSize)

		emit := func(ctx context.Context, dst ro.Observer[[]string]) {
			for i := 0; i < rows; i++ {
				if async && rng.Intn(3) == 0 {
					runtime.Gosched()
				}
				dst.NextWithContext(ctx, []string{big, fmt.Sprint(i)})
			}
			dst.CompleteWithContext(ctx)
		}
		src := ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[[]string]) ro.Teardown {
			if async {
				go emit(ctx, dst)
			} else {
				emit(ctx, dst)
			}
			return nil
		})

		var nexts, errs, completes, afterTerminal int32
		var orderViolation int32
		terminated := make(chan struct{})
		sink := NewCSVWriter(csv.NewWriter(&xfuzz.FailingWriter{FailAt: int(seed&0x7fffffff) % (rows - 1)}))
		sub := sink(src).Subscribe(ro.NewObserver(
			func(int) {
				if atomic.LoadInt32(&errs)+atomic.LoadInt32(&completes) > 0 {
					atomic.AddInt32(&afterTerminal, 1)
				}
				atomic.AddInt32(&nexts, 1)
			},
			func(error) {
				if atomic.AddInt32(&errs, 1) == 1 {
					close(terminated)
				}
			},
			func() {
				atomic.AddInt32(&completes, 1)
				if atomic.LoadInt32(&errs) == 0 {
					// A failing writer must never look like a clean completion.
					atomic.StoreInt32(&orderViolation, 1)
				}
				select {
				case <-terminated:
				default:
					close(terminated)
				}
			},
		))
		defer sub.Unsubscribe()

		xfuzz.WaitChan(t, terminated, "sink termination")
		time.Sleep(20 * time.Millisecond) // lets a duplicate notification from a still-running async source show up

		if n := atomic.LoadInt32(&errs); n != 1 {
			t.Fatalf("async=%v: %d errors, want 1 (completes=%d)", async, n, atomic.LoadInt32(&completes))
		}
		if atomic.LoadInt32(&nexts) != 1 || atomic.LoadInt32(&afterTerminal) != 0 || atomic.LoadInt32(&orderViolation) != 0 {
			t.Fatalf("async=%v: nexts=%d afterTerminal=%d completes=%d", async, atomic.LoadInt32(&nexts), atomic.LoadInt32(&afterTerminal), atomic.LoadInt32(&completes))
		}
	})
}
