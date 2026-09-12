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

package rosimd

import (
	"context"

	"github.com/samber/ro"
)

// ReduceContains collapses a whole stream to a single bool.
//
// It accepts only the Partial types. Answering the question needs the validity mask,
// since padded lanes are zero-filled and would otherwise report a match on a search for
// zero, and the standard library's vector types carry no mask.
//
// The element-wise counterpart is the Contains method on each Partial type, which
// returns a per-lane mask instead.

// ReduceContainsInt8 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt8 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt8[V Int8Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsInt16 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt16 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt16(rosimd.BroadcastInt16(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt16[V Int16Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsInt32 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt32 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt32(rosimd.BroadcastInt32(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt32[V Int32Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsInt64 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastInt64 so every lane of target holds it:
//
//	rosimd.ReduceContainsInt64(rosimd.BroadcastInt64(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsInt64[V Int64Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsUint8 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastUint8 so every lane of target holds it:
//
//	rosimd.ReduceContainsUint8(rosimd.BroadcastUint8(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsUint8[V Uint8Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsUint16 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastUint16 so every lane of target holds it:
//
//	rosimd.ReduceContainsUint16(rosimd.BroadcastUint16(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsUint16[V Uint16Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsUint32 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastUint32 so every lane of target holds it:
//
//	rosimd.ReduceContainsUint32(rosimd.BroadcastUint32(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsUint32[V Uint32Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsUint64 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastUint64 so every lane of target holds it:
//
//	rosimd.ReduceContainsUint64(rosimd.BroadcastUint64(42))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer.
func ReduceContainsUint64[V Uint64Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsFloat32 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastFloat32 so every lane of target holds it:
//
//	rosimd.ReduceContainsFloat32(rosimd.BroadcastFloat32(4.2))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer. Searching for NaN always reports false,
// since NaN equals nothing, not even itself.
func ReduceContainsFloat32[V Float32Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// ReduceContainsFloat64 reports whether any valid lane of the stream matches target.
//
// Widen the value with BroadcastFloat64 so every lane of target holds it:
//
//	rosimd.ReduceContainsFloat64(rosimd.BroadcastFloat64(4.2))
//
// It emits as soon as a match is found rather than waiting for completion, so an
// infinite stream still produces an answer. Searching for NaN always reports false,
// since NaN equals nothing, not even itself.
func ReduceContainsFloat64[V Float64Searchable[V]](target V) func(ro.Observable[V]) ro.Observable[bool] {
	return func(source ro.Observable[V]) ro.Observable[bool] {
		return containsAny(source, target)
	}
}

// LaneMatcher is a vector type that can test its own valid lanes for a value.
//
// Only this package's Partial types satisfy it: answering the question needs the
// validity mask, since padded lanes are zero-filled and would otherwise report a
// false match when searching for zero, and the standard library's vector types carry
// no mask to consult.
//
// The method is unexported deliberately. The exported Contains is element-wise and
// returns a mask; collapsing that mask to one bool is a detail of the Reduce
// operators, and an unexported method also keeps the constraint unsatisfiable from
// outside this package, which matches the fact that only Partial types can implement
// it correctly.
type LaneMatcher[V any] interface {
	anyMatch(V) bool
}

// containsAny reports whether any vector in the stream matches target, emitting as
// soon as one does so an infinite stream still produces an answer.
func containsAny[V LaneMatcher[V]](source ro.Observable[V], target V) ro.Observable[bool] {
	return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[bool]) ro.Teardown {
		found := false

		sub := source.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value V) {
					if found || !value.anyMatch(target) {
						return
					}

					found = true
					destination.NextWithContext(ctx, true)
					destination.CompleteWithContext(ctx)
				},
				destination.ErrorWithContext,
				func(ctx context.Context) {
					if found {
						return
					}

					destination.NextWithContext(ctx, false)
					destination.CompleteWithContext(ctx)
				},
			),
		)

		return sub.Unsubscribe
	})
}
