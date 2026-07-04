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
	"iter"

	"github.com/samber/ro"
)

// ToSeq converts an observable to a Go sequence iterator.
func ToSeq[T any](source ro.Observable[T]) iter.Seq[T] {
	return func(yield func(T) bool) {
		// Create channels for synchronization
		values := make(chan T, 1)
		done := make(chan struct{})

		// Create a context for cancellation
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Subscribe to the observable
		sub := source.SubscribeWithContext(
			ctx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value T) {
					select {
					case values <- value:
					case <-ctx.Done():
					}
				},
				func(ctx context.Context, err error) {
					defer close(done)
					panic(err)
				},
				func(ctx context.Context) {
					close(done)
				},
			),
		)

		// Clean up subscription
		defer sub.Unsubscribe()

		// Yield values as they arrive
		for {
			select {
			case value := <-values:
				if !yield(value) {
					return
				}
			case <-done:
				return
			}
		}
	}
}

// ToSeq2 converts an observable to a Go sequence iterator with index-value pairs.
// Play: https://go.dev/play/p/RGfs_uybXYJ
func ToSeq2[T any](source ro.Observable[T]) iter.Seq2[int, T] {
	return func(yield func(int, T) bool) {
		// Create channels for synchronization
		values := make(chan T, 1)
		done := make(chan struct{})

		// Create a context for cancellation
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		// Subscribe to the observable
		sub := source.SubscribeWithContext(
			ctx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value T) {
					select {
					case values <- value:
					case <-ctx.Done():
					}
				},
				func(ctx context.Context, err error) {
					defer close(done)
					panic(err)
				},
				func(ctx context.Context) {
					close(done)
				},
			),
		)

		// Clean up subscription
		defer sub.Unsubscribe()

		// Yield key-value pairs as they arrive
		i := 0
		for {
			select {
			case value := <-values:
				if !yield(i, value) {
					return
				}
				i++
			case <-done:
				return
			}
		}
	}
}
