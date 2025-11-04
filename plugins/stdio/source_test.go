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
	"errors"
	"strings"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type mockReader struct{}

func (m *mockReader) Read(p []byte) (n int, err error) {
	return 0, errors.New("mock error")
}

func TestNewIOReader(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reader := strings.NewReader("Hello, World!")

	values, err := ro.Collect(
		ro.Pipe1(
			NewIOReader(reader),
			ro.Reduce(
				func(agg []byte, item []byte) []byte {
					return append(agg, item...)
				},
				[]byte{},
			),
		),
	)
	is.Len(values, 1)
	is.Equal([]byte("Hello, World!"), values[0])
	is.Nil(err)
}

func TestNewIOReader_Error(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reader := &mockReader{}

	_, err := ro.Collect(NewIOReader(reader))
	is.NotNil(err)
	is.Equal("mock error", err.Error())
}

func TestNewIOReaderLine(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reader := strings.NewReader("Hello,\nWorld!")

	values, err := ro.Collect(
		NewIOReaderLine(reader),
	)
	is.Len(values, 2)
	is.Equal([]byte("Hello,"), values[0])
	is.Equal([]byte("World!"), values[1])
	is.Nil(err)
}

func TestNewIOReaderLine_Error(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reader := &mockReader{}

	_, err := ro.Collect(NewIOReaderLine(reader))
	is.NotNil(err)
	is.Equal("mock error", err.Error())
}

func TestNewStdReader(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	obs := NewStdReader()
	is.NotNil(obs)
}

func TestNewStdReaderLine(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	obs := NewStdReaderLine()
	is.NotNil(obs)
}

type mockCloserReader struct {
	closed bool
}

func (m *mockCloserReader) Read(p []byte) (n int, err error) {
	return 0, errors.New("mock error")
}

func (m *mockCloserReader) Close() error {
	m.closed = true
	return nil
}

func TestNewIOReader_Closer(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	reader := &mockCloserReader{}
	obs := NewIOReader(reader)

	sub := obs.Subscribe(ro.NoopObserver[[]byte]())
	sub.Unsubscribe()

	is.True(reader.closed)
}
