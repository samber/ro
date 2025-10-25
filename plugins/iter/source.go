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

	"github.com/samber/lo"
	"github.com/samber/ro"
)

// FromSeq creates an observable from a Go sequence iterator.
// Play: https://go.dev/play/p/Cq-cq_AR4Z6
func FromSeq[T any](iterator iter.Seq[T]) ro.Observable[T] {
	return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
		for v := range iterator {
			destination.NextWithContext(subscriberCtx, v)
		}
		destination.CompleteWithContext(subscriberCtx)
		return nil
	})
}

// FromSeq2 creates an observable from a Go sequence iterator with key-value pairs.
// Play: https://go.dev/play/p/d-SZxjCKm9N
func FromSeq2[K, V any](iterator iter.Seq2[K, V]) ro.Observable[lo.Tuple2[K, V]] {
	return ro.NewObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[lo.Tuple2[K, V]]) ro.Teardown {
		for k, v := range iterator {
			destination.NextWithContext(subscriberCtx, lo.T2(k, v))
		}
		destination.CompleteWithContext(subscriberCtx)
		return nil
	})
}
