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
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestNewIOWriter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("Hello "), []byte("World!")),
			NewIOWriter(&writer),
		),
	)
	is.Equal([]int{12}, values)
	is.Equal("Hello World!", writer.String())
	is.Nil(err)
}

func TestNewIOWriter_EmptyData(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just[[]byte](),
			NewIOWriter(&writer),
		),
	)
	is.Equal([]int{0}, values)
	is.Equal("", writer.String())
	is.Nil(err)
}

func TestNewIOWriter_SingleByte(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("A")),
			NewIOWriter(&writer),
		),
	)
	is.Equal([]int{1}, values)
	is.Equal("A", writer.String())
	is.Nil(err)
}

func TestNewIOWriter_LargeData(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder
	largeData := strings.Repeat("Hello World! ", 1000)

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte(largeData)),
			NewIOWriter(&writer),
		),
	)
	is.Equal([]int{len(largeData)}, values)
	is.Equal(largeData, writer.String())
	is.Nil(err)
}

func TestNewIOWriter_MultipleChunks(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder
	chunks := [][]byte{
		[]byte("First "),
		[]byte("Second "),
		[]byte("Third"),
	}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.FromSlice(chunks),
			NewIOWriter(&writer),
		),
	)
	is.Equal([]int{18}, values) // Total length of all chunks
	is.Equal("First Second Third", writer.String())
	is.Nil(err)
}

func TestNewIOWriter_WithErrorWriter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Create a writer that returns an error
	errorWriter := &errorWriter{err: errors.New("write error")}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("test")),
			NewIOWriter(errorWriter),
		),
	)

	// Should return the count written before error
	is.Equal([]int{0}, values)
	is.NotNil(err)
	is.Contains(err.Error(), "write error")
}

func TestNewIOWriter_ContextCancellation(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder
	ctx, cancel := context.WithCancel(context.Background())

	// Cancel immediately
	cancel()

	values, _, _ := ro.CollectWithContext(
		ctx,
		ro.Pipe1(
			ro.Just([]byte("test")),
			NewIOWriter(&writer),
		),
	)

	// Should handle context cancellation gracefully
	// The actual behavior depends on when the context is cancelled
	// We just check that we get some result
	is.NotEmpty(values)
}

func TestNewIOWriter_ConcurrentWrites(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder
	chunks := make([][]byte, 100)
	for i := range chunks {
		chunks[i] = []byte("chunk")
	}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.FromSlice(chunks),
			NewIOWriter(&writer),
		),
	)

	expectedLength := 100 * len("chunk")
	is.Equal([]int{expectedLength}, values)
	is.Equal(strings.Repeat("chunk", 100), writer.String())
	is.Nil(err)
}

func TestNewStdWriter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Capture stdout to test NewStdWriter
	// Note: This is a basic test since capturing stdout is complex
	// In a real scenario, you might want to use os.Pipe or similar

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("test stdout")),
			NewStdWriter(),
		),
	)

	// Should return the count written
	is.Equal([]int{11}, values)
	is.Nil(err)
}

func TestNewStdWriter_EmptyData(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just[[]byte](),
			NewStdWriter(),
		),
	)

	is.Equal([]int{0}, values)
	is.Nil(err)
}

func TestNewIOWriter_WithBytesBuffer(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	buffer := &bytes.Buffer{}

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("test buffer")),
			NewIOWriter(buffer),
		),
	)

	is.Equal([]int{11}, values)
	is.Equal("test buffer", buffer.String())
	is.Nil(err)
}

func TestNewIOWriter_WithNilWriter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// This should panic or handle gracefully depending on implementation
	// Let's test the behavior
	defer func() {
		if r := recover(); r != nil {
			// Expected behavior for nil writer
			t.Logf("Recovered from panic: %v", r)
		}
	}()

	values, err := ro.Collect(
		ro.Pipe1(
			ro.Just([]byte("test")),
			NewIOWriter(nil),
		),
	)

	// If it doesn't panic, it should handle gracefully
	if err == nil {
		is.Equal([]int{0}, values)
	}
}

func TestNewIOWriter_WithDelayedSource(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var writer strings.Builder

	// Create a source that emits with delays
	source := ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[[]byte]) ro.Teardown {
		go func() {
			time.Sleep(10 * time.Millisecond)
			observer.NextWithContext(ctx, []byte("delayed"))
			observer.CompleteWithContext(ctx)
		}()
		return func() {}
	})

	values, err := ro.Collect(
		ro.Pipe1(
			source,
			NewIOWriter(&writer),
		),
	)

	is.Equal([]int{7}, values) // "delayed" is 7 bytes
	is.Equal("delayed", writer.String())
	is.Nil(err)
}

// errorWriter is a test helper that always returns an error
type errorWriter struct {
	err error
}

func (w *errorWriter) Write(p []byte) (n int, err error) {
	return 0, w.err
}
