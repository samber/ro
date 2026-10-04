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
	"bytes"
	"io"
	"math/rand"
	"os"
	"runtime"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

func waitDone(t *testing.T, ch <-chan struct{}, what string) {
	t.Helper()
	select {
	case <-ch:
	case <-time.After(fuzzWait):
		t.Fatalf("%s: timed out after %s", what, fuzzWait)
	}
}

// assertNoLeak fails when goroutines started by the scenario are still alive after a bounded settle loop.
func assertNoLeak(t *testing.T, before int) {
	t.Helper()
	deadline := time.Now().Add(fuzzSettle)
	for runtime.NumGoroutine() > before {
		if time.Now().After(deadline) {
			t.Fatalf("goroutine leak: before=%d after=%d", before, runtime.NumGoroutine())
		}
		time.Sleep(5 * time.Millisecond)
	}
}

// countingInfiniteReader never blocks and never ends (until fuzzReadCap), repeating pattern.
type countingInfiniteReader struct {
	pattern []byte
	reads   int64
}

func (r *countingInfiniteReader) Read(p []byte) (int, error) {
	if atomic.AddInt64(&r.reads, 1) > fuzzReadCap {
		return 0, io.EOF
	}
	return copy(p, bytes.Repeat(r.pattern, 1+len(p)/len(r.pattern))), nil
}

// FuzzIOReaderStopsAfterTake checks that the read loop ends once downstream completed early.
// Sync: a reader that never blocks. Async: a pipe fed by a goroutine that then idles without closing it,
// like a terminal or a socket waiting for more input.
func FuzzIOReaderStopsAfterTake(f *testing.F) {
	f.Skip("race: stdio-reader-ignores-downstream-close; remove when fixed")
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		lineMode := mask&(1<<1) != 0
		take := 1 + int64(mask>>2)%fuzzMaxItems
		before := runtime.NumGoroutine()

		var reader io.Reader
		var inf *countingInfiniteReader
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
					if _, err := pw.Write([]byte("hello\n")); err != nil {
						return
					}
				}
				<-stop
			}()
		} else {
			inf = &countingInfiniteReader{pattern: []byte("hello\n")}
			reader = inf
		}
		t.Cleanup(release)

		source := NewIOReader(reader)
		if lineMode {
			source = NewIOReaderLine(reader)
		}

		var received int64
		completed := make(chan struct{})
		returned := make(chan struct{})
		go func() {
			defer close(returned)
			sub := ro.Take[[]byte](take)(source).Subscribe(ro.NewObserver(
				func([]byte) { atomic.AddInt64(&received, 1) },
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
		release()
		assertNoLeak(t, before)
	})
}

// FuzzIOReaderEmitsStableBuffers checks that emitted slices stay valid after the reader overwrote its read buffer,
// which an asynchronous downstream (ObserveOn) only consumes later.
func FuzzIOReaderEmitsStableBuffers(f *testing.F) {
	f.Skip("race: stdio-reader-shared-buffer; remove when fixed")
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		chunks := 2 + int(mask>>1)%fuzzMaxItems
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving

		// Chunks differ from one another so an overwritten buffer is detectable.
		want := make([][]byte, chunks)
		for i := range want {
			want[i] = bytes.Repeat([]byte{byte('a' + i%26)}, 1+rng.Intn(32))
		}

		var reader io.Reader
		if async {
			pr, pw := io.Pipe()
			reader = pr
			go func() {
				for _, c := range want {
					if rng.Intn(3) == 0 {
						runtime.Gosched()
					}
					if _, err := pw.Write(c); err != nil {
						return
					}
				}
				_ = pw.Close()
			}()
		} else {
			reader = &chunkReader{chunks: want}
		}

		got, err := ro.Collect(ro.ObserveOn[[]byte](chunks + 1)(NewIOReader(reader)))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(got) != chunks {
			t.Fatalf("async=%v: got %d chunks, want %d", async, len(got), chunks)
		}
		for i := range want {
			if !bytes.Equal(got[i], want[i]) {
				t.Fatalf("async=%v: chunk %d is %q, want %q", async, i, got[i], want[i])
			}
		}
	})
}

// chunkReader returns one chunk per Read, so chunk boundaries are deterministic.
type chunkReader struct {
	chunks [][]byte
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if len(r.chunks) == 0 {
		return 0, io.EOF
	}
	n := copy(p, r.chunks[0])
	r.chunks = r.chunks[1:]
	return n, nil
}

// eofChunk is one Read result: data, optionally delivered together with io.EOF.
type eofChunk struct {
	data []byte
	last bool
}

// eofWithDataReader returns its final chunk together with io.EOF, which io.Reader explicitly allows.
// Chunks come from a channel so an async producer can hand them over at its own pace.
type eofWithDataReader struct {
	feed <-chan eofChunk
}

func (r *eofWithDataReader) Read(p []byte) (int, error) {
	chunk, ok := <-r.feed
	if !ok {
		return 0, io.EOF
	}
	n := copy(p, chunk.data)
	if chunk.last {
		return n, io.EOF
	}
	return n, nil
}

// FuzzIOReaderKeepsDataReturnedWithEOF checks that bytes returned together with io.EOF are emitted.
func FuzzIOReaderKeepsDataReturnedWithEOF(f *testing.F) {
	f.Skip("race: stdio-reader-drops-final-bytes-with-eof; remove when fixed")
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0
		chunks := 1 + int(mask>>1)%fuzzMaxItems
		rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving

		want := make([][]byte, chunks)
		var wantAll []byte
		for i := range want {
			want[i] = bytes.Repeat([]byte{byte('a' + i%26)}, 1+rng.Intn(16))
			wantAll = append(wantAll, want[i]...)
		}

		feed := make(chan eofChunk, chunks)
		produce := func() {
			for i, c := range want {
				if async && rng.Intn(3) == 0 {
					runtime.Gosched()
				}
				feed <- eofChunk{data: c, last: i == len(want)-1}
			}
			close(feed)
		}
		if async {
			go produce()
		} else {
			produce()
		}
		reader := &eofWithDataReader{feed: feed}

		got, err := ro.Collect(NewIOReader(reader))
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		var gotAll []byte
		for _, c := range got {
			gotAll = append(gotAll, c...)
		}
		if !bytes.Equal(gotAll, wantAll) {
			t.Fatalf("async=%v: received %d bytes %q, want %d bytes %q", async, len(gotAll), gotAll, len(wantAll), wantAll)
		}
	})
}

// FuzzPromptStopsAfterTake checks that NewPrompt stops reading stdin once downstream completed early.
// Sync: the line is already in the pipe at subscription time. Async: it arrives after a goroutine delay.
func FuzzPromptStopsAfterTake(f *testing.F) {
	f.Skip("race: stdio-prompt-ignores-downstream-close; remove when fixed")
	fuzzSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		async := mask&maskAsync != 0

		stdinR, stdinW, err := os.Pipe()
		if err != nil {
			t.Fatalf("pipe: %v", err)
		}
		devNull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
		if err != nil {
			t.Fatalf("open devnull: %v", err)
		}
		returned := make(chan struct{})
		origIn, origOut := os.Stdin, os.Stdout
		os.Stdin, os.Stdout = stdinR, devNull
		t.Cleanup(func() {
			// Closing stdin ends a prompt loop that is still alive; wait for it before restoring the
			// globals it reads, so a leaked loop cannot race with the restore.
			_ = stdinW.Close()
			select {
			case <-returned:
			case <-time.After(fuzzSettle):
			}
			os.Stdin, os.Stdout = origIn, origOut
			_ = stdinR.Close()
			_ = devNull.Close()
		})

		if async {
			rng := rand.New(rand.NewSource(seed)) //nolint:gosec // deterministic interleaving
			go func() {
				time.Sleep(time.Duration(rng.Intn(5)) * time.Millisecond)
				_, _ = stdinW.WriteString("hello\n")
			}()
		} else {
			_, _ = stdinW.WriteString("hello\n")
		}

		completed := make(chan struct{})
		go func() {
			defer close(returned)
			sub := ro.Take[[]byte](1)(NewPrompt("> ")).Subscribe(ro.NewObserver(
				func([]byte) {},
				func(error) {},
				func() { close(completed) },
			))
			defer sub.Unsubscribe()
		}()

		waitDone(t, completed, "Take(1) completion")
		waitDone(t, returned, "prompt loop stop after Take(1)")
	})
}
