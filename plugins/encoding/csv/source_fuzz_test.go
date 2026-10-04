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
	"bytes"
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/samber/ro"
)

// infiniteCSV never blocks and yields "a,b" records forever, until fuzzReadCap reads.
type infiniteCSV struct {
	reads int64
}

func (r *infiniteCSV) Read(p []byte) (int, error) {
	if atomic.AddInt64(&r.reads, 1) > fuzzReadCap {
		return 0, io.EOF
	}
	return copy(p, bytes.Repeat([]byte("a,b\n"), 1+len(p)/4)), nil
}

// FuzzCSVReaderStopsAfterTake checks that reading stops once downstream completed early.
// Sync: a reader that never blocks. Async: a pipe whose writer sends the wanted rows and then idles without closing.
func FuzzCSVReaderStopsAfterTake(f *testing.F) {
	f.Skip("race: csv-reader-ignores-downstream-close; remove when fixed")
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		take := 1 + int64(mask>>1)%fuzzMaxItems

		var reader io.Reader
		var inf *infiniteCSV
		release := func() {}
		if async {
			pr, pw := io.Pipe()
			reader = pr
			stop := make(chan struct{})
			release = func() { close(stop); _ = pw.Close() }
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving
			go func() {
				for i := int64(0); i < take; i++ {
					if rng.Intn(3) == 0 {
						runtime.Gosched()
					}
					if _, err := pw.Write([]byte("a,b\n")); err != nil {
						return
					}
				}
				<-stop
			}()
		} else {
			inf = &infiniteCSV{}
			reader = inf
		}
		t.Cleanup(release)

		var received int64
		completed := make(chan struct{})
		returned := make(chan struct{})
		go func() {
			defer close(returned)
			sub := ro.Take[[]string](take)(NewCSVReader(csv.NewReader(reader))).Subscribe(ro.NewObserver(
				func([]string) { atomic.AddInt64(&received, 1) },
				func(error) {},
				func() { close(completed) },
			))
			defer sub.Unsubscribe()
		}()

		waitDone(t, completed, "Take completion")
		waitDone(t, returned, "read loop stop after Take")

		if inf != nil && atomic.LoadInt64(&inf.reads) >= fuzzReadCap {
			t.Fatalf("reader was drained to its cap (%d reads) for Take(%d)", fuzzReadCap, take)
		}
		if got := atomic.LoadInt64(&received); got != take {
			t.Fatalf("received %d rows, want %d", got, take)
		}
	})
}

// FuzzCSVReaderKeepsEveryRow checks that no row is lost, duplicated or reordered when a slow async consumer
// (ObserveOn) drains the stream, with a sync (in-memory) or async (pipe) reader.
func FuzzCSVReaderKeepsEveryRow(f *testing.F) {
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		rows := 1 + int(mask>>1)%fuzzMaxItems*4
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving

		want := make([][]string, rows)
		var doc strings.Builder
		for i := range want {
			want[i] = []string{fmt.Sprint(i), strings.Repeat("v", 1+rng.Intn(8))}
			doc.WriteString(want[i][0] + "," + want[i][1] + "\n")
		}

		var reader io.Reader = strings.NewReader(doc.String())
		if async {
			pr, pw := io.Pipe()
			reader = pr
			content := doc.String()
			go func() {
				for len(content) > 0 {
					n := 1 + rng.Intn(7)
					if n > len(content) {
						n = len(content)
					}
					if _, err := pw.Write([]byte(content[:n])); err != nil {
						return
					}
					content = content[n:]
					if rng.Intn(3) == 0 {
						runtime.Gosched()
					}
				}
				_ = pw.Close()
			}()
		}

		got, err := ro.Collect(ro.ObserveOn[[]string](rows + 1)(NewCSVReader(csv.NewReader(reader))))
		if err != nil {
			t.Fatalf("async=%v: unexpected error: %v", async, err)
		}
		if len(got) != rows {
			t.Fatalf("async=%v: got %d rows, want %d", async, len(got), rows)
		}
		for i := range want {
			if len(got[i]) != 2 || got[i][0] != want[i][0] || got[i][1] != want[i][1] {
				t.Fatalf("async=%v: row %d is %v, want %v", async, i, got[i], want[i])
			}
		}
	})
}
