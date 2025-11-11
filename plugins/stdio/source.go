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
	"bufio"
	"context"
	"io"
	"os"

	"github.com/samber/ro"
)

const IOReaderBufferSize = 1024

// NewIOReader creates an observable that reads bytes from an io.Reader.
// Play: https://go.dev/play/p/b75Poy3EVYn
func NewIOReader(reader io.Reader) ro.Observable[[]byte] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[[]byte]) ro.Teardown {
		buf := make([]byte, IOReaderBufferSize)

		for {
			n, err := reader.Read(buf)
			if err != nil {
				if err == io.EOF {
					destination.CompleteWithContext(ctx)
				} else {
					destination.ErrorWithContext(ctx, err)
				}
				break
			}
			destination.NextWithContext(ctx, buf[:n])
		}

		return func() {
			if closer, ok := reader.(io.Closer); ok {
				closer.Close()
			}
		}
	})
}

// NewIOReaderLine creates an observable that reads lines from an io.Reader.
// Play: https://go.dev/play/p/oMv2jYVSLqd
func NewIOReaderLine(reader io.Reader) ro.Observable[[]byte] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[[]byte]) ro.Teardown {
		r := bufio.NewReader(reader)

		for {
			lines, _, err := r.ReadLine()
			if err != nil {
				if err == io.EOF {
					destination.CompleteWithContext(ctx)
				} else {
					destination.ErrorWithContext(ctx, err)
				}
				break
			}

			output := make([]byte, len(lines))
			copy(output, lines)
			destination.NextWithContext(ctx, output)
		}

		return func() {
			if closer, ok := reader.(io.Closer); ok {
				closer.Close()
			}
		}
	})
}

// NewStdReader creates an observable that reads bytes from standard input.
func NewStdReader() ro.Observable[[]byte] {
	return NewIOReader(os.Stdin)
}

// NewStdReaderLine creates an observable that reads lines from standard input.
func NewStdReaderLine() ro.Observable[[]byte] {
	return NewIOReaderLine(os.Stdin)
}

// NewPrompt creates an observable that reads user input after displaying a prompt.
func NewPrompt(prompt string) ro.Observable[[]byte] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[[]byte]) ro.Teardown {
		for {
			// Print the prompt to stdout
			os.Stdout.WriteString(prompt)

			// Read from stdin
			reader := bufio.NewReader(os.Stdin)
			line, _, err := reader.ReadLine()
			if err != nil {
				if err == io.EOF {
					break
				} else {
					destination.ErrorWithContext(ctx, err)
					return func() {}
				}
			}

			// Send the input as a byte slice
			destination.NextWithContext(ctx, line)
		}

		destination.CompleteWithContext(ctx)

		return func() {}
	})
}
