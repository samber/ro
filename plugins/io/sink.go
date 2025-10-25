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


package roio

import (
	"context"
	"io"
	"os"

	"github.com/samber/ro"
)

// NewIOWriter creates a sink that writes byte slices to an io.Writer and emits the total bytes written.
// Play: https://go.dev/play/p/XoLdEcsmKxU
func NewIOWriter(writer io.Writer) func(ro.Observable[[]byte]) ro.Observable[int] {
	return func(source ro.Observable[[]byte]) ro.Observable[int] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[int]) ro.Teardown {
			count := 0

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value []byte) {
						n, err := writer.Write(value)
						if err != nil {
							destination.NextWithContext(ctx, count)
							destination.ErrorWithContext(ctx, err)
						} else {
							count += n
						}
					},
					func(ctx context.Context, err error) {
						destination.NextWithContext(ctx, count)
						destination.ErrorWithContext(ctx, err)
					},
					func(ctx context.Context) {
						destination.NextWithContext(ctx, count)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}

// NewStdWriter creates a sink that writes byte slices to standard output and emits the total bytes written.
// Play: https://go.dev/play/p/9GjhDJIAs7z
func NewStdWriter() func(ro.Observable[[]byte]) ro.Observable[int] {
	return func(source ro.Observable[[]byte]) ro.Observable[int] {
		return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[int]) ro.Teardown {
			count := 0

			sub := source.SubscribeWithContext(
				subscriberCtx,
				ro.NewObserverWithContext(
					func(ctx context.Context, value []byte) {
						n, err := os.Stdout.Write(value)
						if err != nil {
							destination.NextWithContext(ctx, count)
							_, _ = os.Stderr.Write([]byte(err.Error()))
							destination.ErrorWithContext(ctx, err)
						} else {
							count += n
						}
					},
					func(ctx context.Context, err error) {
						destination.NextWithContext(ctx, count)
						_, _ = os.Stderr.Write([]byte(err.Error()))
						destination.ErrorWithContext(ctx, err)
					},
					func(ctx context.Context) {
						destination.NextWithContext(ctx, count)
						destination.CompleteWithContext(ctx)
					},
				),
			)

			return sub.Unsubscribe
		})
	}
}
