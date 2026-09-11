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
	"simd"

	"github.com/samber/ro"
)

// The midway compiler pass hard-codes the identifier `simd` when it injects
// "simd/archsimd" into any file holding simd-dependent code, so every file in this
// package imports simd. The functions in this file are generic and never touch a
// simd type themselves, which is why a package-level reference is enough here.
var _ = simd.BroadcastInt8s

// maxLanes is the widest vector any supported architecture can produce, in 8-bit
// lanes (a 512-bit register). Stack buffers in the per-type files are sized to this
// so they never need to grow, whatever width the specializer picks.
const maxLanes = 64

// mapVector applies a lane-wise operation to every vector in the stream.
//
// The body only ever calls the transform it is given, never a simd function.
// Reaching a simd package function from a generic function — even indirectly through
// a helper — makes it simd-dependent and breaks the dispatcher the compiler
// generates between it and its per-width clones.
func mapVector[V any](source ro.Observable[V], transform func(V) V) ro.Observable[V] {
	return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[V]) ro.Teardown {
		sub := source.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value V) {
					destination.NextWithContext(ctx, transform(value))
				},
				destination.ErrorWithContext,
				destination.CompleteWithContext,
			),
		)

		return sub.Unsubscribe
	})
}

// zipVector combines two vector streams lane-wise, in lockstep.
//
// Values arriving before their counterpart are held, so a slow side never drops
// data. The stream ends as soon as either side completes and its buffer is drained,
// matching ro.ZipWith.
func zipVector[V any](source, other ro.Observable[V], combine func(V, V) V) ro.Observable[V] {
	return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[V]) ro.Teardown {
		var (
			pendingLeft  []V
			pendingRight []V
			leftDone     bool
			rightDone    bool
			done         bool
		)

		// A side finishing is not enough to finish the zip: a synchronous cold source
		// drains completely before the other side is even subscribed, so the first
		// completion routinely arrives with a full buffer still waiting for a partner.
		// The zip ends only once a side is both complete and drained, because only
		// then can no further pair be formed.
		maybeComplete := func(ctx context.Context) {
			if done {
				return
			}

			if (leftDone && len(pendingLeft) == 0) || (rightDone && len(pendingRight) == 0) {
				done = true
				destination.CompleteWithContext(ctx)
			}
		}

		emit := func(ctx context.Context) {
			for len(pendingLeft) > 0 && len(pendingRight) > 0 {
				left, right := pendingLeft[0], pendingRight[0]
				pendingLeft, pendingRight = pendingLeft[1:], pendingRight[1:]

				destination.NextWithContext(ctx, combine(left, right))
			}

			maybeComplete(ctx)
		}

		leftSub := source.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value V) {
					pendingLeft = append(pendingLeft, value)
					emit(ctx)
				},
				destination.ErrorWithContext,
				func(ctx context.Context) {
					leftDone = true
					maybeComplete(ctx)
				},
			),
		)

		rightSub := other.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value V) {
					pendingRight = append(pendingRight, value)
					emit(ctx)
				},
				destination.ErrorWithContext,
				func(ctx context.Context) {
					rightDone = true
					maybeComplete(ctx)
				},
			),
		)

		return func() {
			leftSub.Unsubscribe()
			rightSub.Unsubscribe()
		}
	})
}

// LaneStore is the part of a vector type that reports its width and hands its valid
// lanes back as scalars.
//
// Len reports lane capacity, not how many lanes hold data. It is meaningful on the
// zero value, which is how the operators discover the architecture's width, and it
// becomes a compile-time constant once the specializer has run.
type LaneStore[T any] interface {
	Len() int
	StorePart(dst []T) int
}

// LaneCount is a vector that knows how many of its lanes hold data.
//
// Only this package's Partial types satisfy it. A standard library vector carries no
// validity mask, so it has no count distinct from its capacity.
type LaneCount interface {
	Count() int
}

// LaneBuffer is a LaneStore that can also be built from a slice of scalars.
//
// Only this package's Partial types satisfy it. The standard library's vector types
// expose no constructor method, and an interface cannot reach the simd.LoadXxx
// package functions, which is why the Vectorize operators always produce Partials.
type LaneBuffer[V any, T any] interface {
	LaneStore[T]

	Load(src []T) V
	LoadPart(src []T, n int) V
}

// vectorize batches a scalar stream into vectors of the width the running
// architecture supports, emitting a partial vector on completion for the remainder.
//
// Construction goes through methods on the type parameter rather than closures,
// because a closure built here would call a simd package function from inside a
// generic body, which breaks the compiler's clone dispatcher.
func vectorize[T any, V LaneBuffer[V, T]](source ro.Observable[T]) ro.Observable[V] {
	return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[V]) ro.Teardown {
		var proto V

		lanes := proto.Len()
		buffer := make([]T, 0, lanes)

		// The trailing vector is emitted from Complete, which carries the
		// subscription context rather than any item's. Remembering the last item's
		// context keeps the tail on the same chain as the values it holds.
		lastCtx := subscriberCtx

		sub := source.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value T) {
					lastCtx = ctx

					buffer = append(buffer, value)
					if len(buffer) == lanes {
						destination.NextWithContext(ctx, proto.Load(buffer))
						buffer = buffer[:0]
					}
				},
				destination.ErrorWithContext,
				func(ctx context.Context) {
					if len(buffer) > 0 {
						destination.NextWithContext(lastCtx, proto.LoadPart(buffer, len(buffer)))
						buffer = buffer[:0]
					}

					destination.CompleteWithContext(ctx)
				},
			),
		)

		return sub.Unsubscribe
	})
}

// reduceLanes folds every valid lane of every vector into a single value.
//
// simd offers no horizontal reduction and no way to extract a mask as a bitmask, so
// collapsing lanes always means storing the vector and looping in scalar code. Doing
// that per vector, rather than keeping a vector accumulator, also keeps the result
// correct for a stream whose vectors are not in vectorize's order — a partial vector
// arriving before a full one would corrupt an accumulator seeded from it.
func reduceLanes[T any, V LaneStore[T]](
	source ro.Observable[V],
	accumulate func(acc, lane T) T,
	onComplete func(acc T, seen bool, emit func(T)),
) ro.Observable[T] {
	return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[T]) ro.Teardown {
		var (
			acc  T
			seen bool
		)

		lastCtx := subscriberCtx

		sub := source.SubscribeWithContext(
			subscriberCtx,
			ro.NewObserverWithContext(
				func(ctx context.Context, value V) {
					lastCtx = ctx

					buffer := make([]T, maxLanes)
					n := value.StorePart(buffer)

					for i := range n {
						if !seen {
							acc, seen = buffer[i], true

							continue
						}

						acc = accumulate(acc, buffer[i])
					}
				},
				destination.ErrorWithContext,
				func(ctx context.Context) {
					onComplete(acc, seen, func(result T) {
						destination.NextWithContext(lastCtx, result)
					})

					destination.CompleteWithContext(ctx)
				},
			),
		)

		return sub.Unsubscribe
	})
}

// emitWhenSeen is the completion rule for reductions with no meaningful value for an
// empty stream, such as min and max: emit nothing rather than a zero that would be
// indistinguishable from a real result. It matches ro.Min.
func emitWhenSeen[T any](acc T, seen bool, emit func(T)) {
	if seen {
		emit(acc)
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
