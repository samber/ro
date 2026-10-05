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

package ro

import (
	"context"
	"math"
	"time"

	"github.com/samber/lo"
	"github.com/samber/ro/internal/xrand"
)

// Of creates an Observable that emits some values you specify.
// Play: https://go.dev/play/p/Zp5LgHgvJ59
func Of[T any](values ...T) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		for _, v := range values {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, v)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Just is an alias for Of.
// Play: https://go.dev/play/p/A5S2McqqfqE
func Just[T any](values ...T) Observable[T] {
	return Of(values...)
}

// Start creates an Observable that emits lazily a single value.
// Play: https://go.dev/play/p/Jz7oyagu07u
func Start[T any](cb func() T) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		destination.NextWithContext(ctx, cb())
		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Timer creates an Observable that emits a value after a specified duration.
// Play: https://go.dev/play/p/hMkNLEqpcy3
func Timer(duration time.Duration) Observable[time.Duration] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[time.Duration]) Teardown {
		timer := time.NewTimer(duration)
		// Release the timer on every path, including context cancellation.
		defer timer.Stop()

		select {
		case <-timer.C:
			destination.NextWithContext(ctx, duration)
			destination.CompleteWithContext(ctx)
		case <-ctx.Done():
			destination.ErrorWithContext(ctx, ctx.Err())
		}

		return nil
	})
}

// Interval creates an Observable that emits an infinite sequence of ascending
// integers, with a constant interval between them. The first value is not emitted
// immediately, but after the first interval has passed.
// Play: https://go.dev/play/p/7yskMPPFHA7
func Interval(interval time.Duration) Observable[int64] {
	return NewObservableWithContext(func(ctx context.Context, destination Observer[int64]) Teardown {
		ticker := time.NewTicker(interval)
		done := make(chan struct{})

		go recoverUnhandledError(func() {
			defer destination.CompleteWithContext(ctx)
			value := int64(0)

			for {
				select {
				case <-done:
					return
				case <-ctx.Done():
					return
				case _, ok := <-ticker.C:
					// `ok` is not expected to be false, because the go runtime will close the channel itself
					if ok {
						destination.NextWithContext(ctx, value)
						value++
					}
				}
			}
		})

		return func() {
			ticker.Stop()
			close(done)
		}
	})
}

// IntervalWithInitial creates an Observable that emits ascending integers starting at zero.
// The first value is emitted after initial, then subsequent values are emitted every interval.
// When initial is zero, the first value is emitted synchronously during subscription.
// Play: https://go.dev/play/p/Xhi6c336ldy
func IntervalWithInitial(initial, interval time.Duration) Observable[int64] {
	return NewObservableWithContext(func(ctx context.Context, destination Observer[int64]) Teardown {
		ticker := time.NewTicker(interval)
		ticker.Stop()
		timer := time.NewTimer(initial)
		done := make(chan struct{}, 1)

		value := int64(0)

		// Synchronous initial value when first tick must be triggered immediately.
		if initial == 0 {
			destination.NextWithContext(ctx, value)

			value++

			ticker.Reset(interval)
		}

		go recoverUnhandledError(func() {
			defer destination.CompleteWithContext(ctx)

			for {
				select {
				case <-done:
					return
				case <-ctx.Done():
					return
				case _, ok := <-timer.C:
					// `ok` is not expected to be false, because the go runtime will close the channel itself
					if ok && initial != 0 { // exclude initial tick when it is immediately
						destination.NextWithContext(ctx, value)
						value++

						ticker.Reset(interval)
					}
				case _, ok := <-ticker.C:
					// `ok` is not expected to be false, because the go runtime will close the channel itself
					if ok {
						destination.NextWithContext(ctx, value)
						value++
					}
				}
			}
		})

		return func() {
			ticker.Stop()
			timer.Stop()
			close(done)
		}
	})
}

// Range creates an Observable that emits a range of integers.
// The range is [start:end), so `start` is emitted but not `end`.
// If `start` is equal to `end`, an empty Observable is returned.
// If `start` is greater than `end`, the emitted values are in
// descending order. The step is 1.
// Play: https://go.dev/play/p/5XAXfNrtJm2
func Range(start, end int64) Observable[int64] {
	sign := int64(1)

	if start == end {
		return Empty[int64]()
	} else if start > end {
		sign = -1
	}

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int64]) Teardown {
		for cursor := start; rangeHasNext(cursor, end, sign); cursor += sign {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, cursor)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// rangeHasNext reports whether cursor has not reached end yet, walking in the direction of sign.
// It compares directly instead of multiplying by sign, because end*sign overflows for math.MinInt64.
func rangeHasNext(cursor, end, sign int64) bool {
	if sign > 0 {
		return cursor < end
	}

	return cursor > end
}

// float64Epsilon is the machine epsilon of float64 (2^-52): the relative spacing of float64
// values around 1. A decimal literal such as 0.07 is stored with a relative error up to half of it.
const float64Epsilon = 0x1p-52

// rangeWithStepEpsilonFactor leaves room for the roundings of the subtraction and the division,
// on top of the representation error of start, end and step.
const rangeWithStepEpsilonFactor = 4

// rangeWithStepEpsilon returns the rounding error to absorb in (end-start)/step. Without it, a
// quotient that should be an exact integer can land just above it, and ceil rounds up to one
// value too many: 0.07/0.01 = 7.000000000000001 would emit 8 values instead of 7.
// The error scales with (|start|+|end|)/step, which also covers the cancellation in end-start
// when both bounds are large and close. A fixed tolerance would be too loose for small ranges
// and too tight for large offsets or large quotients.
func rangeWithStepEpsilon(start, end, step float64) float64 {
	return rangeWithStepEpsilonFactor * float64Epsilon * (math.Abs(start) + math.Abs(end)) / step
}

// rangeWithStepCount returns the number of values in [start:end) walked by step.
// It is shared by RangeWithStep and RangeWithStepAndInterval, so both always emit
// the same number of values. start and end must differ and step must be positive.
func rangeWithStepCount(start, end, step float64) int64 {
	count := int64(math.Ceil(math.Abs(end-start)/step - rangeWithStepEpsilon(start, end, step)))

	// start differs from end, so the range always contains at least `start`.
	if count < 1 {
		return 1
	}

	return count
}

// RangeWithStep creates an Observable that emits a range of floats.
// The range is [start:end), so `start` is emitted but not `end`.
// If `start` is equal to `end`, an empty Observable is returned.
// If `start` is greater than `end`, the emitted values are in
// descending order.
// The step must be greater than 0.
// Play: https://go.dev/play/p/EOG0tIVjUKC
func RangeWithStep(start, end, step float64) Observable[float64] {
	sign := 1.0

	if start == end {
		return Empty[float64]()
	} else if start > end {
		sign = -1.0
	}

	if step <= 0 {
		panic(ErrRangeWithStepWrongStep)
	}

	count := rangeWithStepCount(start, end, step)

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[float64]) Teardown {
		for i := int64(0); i < count; i++ {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, rangeWithStepValue(start, step, sign, i))
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// rangeWithStepValue returns the i-th value of the range. It multiplies instead of
// accumulating step, so rounding error does not grow with i.
func rangeWithStepValue(start, step, sign float64, i int64) float64 {
	return start + float64(i)*sign*step
}

// RangeWithInterval creates an Observable that emits a range of integers.
// The range is [start:end), so `start` is emitted but not `end`.
// If `start` is equal to `end`, an empty Observable is returned.
// If `start` is greater than `end`, the emitted values are in
// descending order. The interval is the time between each value.
// The first value is emitted after the first interval has passed.
// The step is 1.
// Play: https://go.dev/play/p/Y_1l6BDbMSi
func RangeWithInterval(start, end int64, interval time.Duration) Observable[int64] {
	sign := int64(1)

	if start == end {
		return Empty[int64]()
	} else if start > end {
		sign = -1
	}

	return Pipe2(
		Interval(interval),
		Map(func(v int64) int64 {
			if start < end {
				return start + v
			}

			return start - v
		}),
		Take[int64]((end*sign)-(start*sign)),
	)
}

// RangeWithStepAndInterval creates an Observable that emits a range of floats.
// The range is [start:end), so `start` is emitted but not `end`.
// If `start` is equal to `end`, an empty Observable is returned.
// If `start` is greater than `end`, the emitted values are in
// descending order. The step must be greater than 0.
// The interval is the time between each value.
// The first value is emitted after the first interval has passed.
// Play: https://go.dev/play/p/kdAEsGwfqw9
func RangeWithStepAndInterval(start, end, step float64, interval time.Duration) Observable[float64] {
	sign := 1.0

	if start == end {
		return Empty[float64]()
	} else if start > end {
		sign = -1.0
	}

	if step <= 0 {
		panic(ErrRangeWithStepAndIntervalWrongStep)
	}

	return Pipe2(
		Interval(interval),
		Map(func(v int64) float64 {
			return rangeWithStepValue(start, step, sign, v)
		}),
		Take[float64](rangeWithStepCount(start, end, step)),
	)
}

// Repeat creates an Observable that emits a single value multiple times.
// This is a creation operator. The pipeable equivalent is `RepeatWith`.
// Play: https://go.dev/play/p/CUvh_TYALNe
func Repeat[T any](item T, count int64) Observable[T] {
	if count < 0 {
		panic(ErrRepeatWrongCount)
	} else if count == 0 {
		return Empty[T]()
	}

	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		for i := int64(0); i < count; i++ {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, item)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// RepeatWithInterval creates an Observable that emits a single value multiple times.
// The interval is the time between each value. The first value is emitted
// after the first interval has passed.
// Play: https://go.dev/play/p/4PK5Zt2sGze
func RepeatWithInterval[T any](item T, count int64, interval time.Duration) Observable[T] {
	if count < 0 {
		panic(ErrRepeatWithIntervalWrongCount)
	} else if count == 0 {
		return Empty[T]()
	}

	return Pipe1(
		RangeWithInterval(0, count, interval),
		Map(func(_ int64) T {
			return item
		}),
	)
}

// FromChannel creates an Observable from a channel. Closing the
// channel will complete the Observable.
// Play: https://go.dev/play/p/x0u4eaOzYln
func FromChannel[T any](in <-chan T) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		done := make(chan struct{})

		go recoverUnhandledError(func() {
			for {
				select {
				case item, ok := <-in:
					if !ok {
						destination.CompleteWithContext(ctx)
						return
					}

					destination.NextWithContext(ctx, item)
				case <-done:
					return
				}
			}
		})

		return func() {
			close(done)
		}
	})
}

// FromSlice creates an Observable from a slice. The values are emitted
// in the order they are in the slice.
// Play: https://go.dev/play/p/BNhnqoQn0tP
func FromSlice[T any](collections ...[]T) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		for _, collection := range collections {
			for _, value := range collection {
				if destination.IsClosed() {
					return nil
				}

				destination.NextWithContext(ctx, value)
			}
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Empty creates an Observable that emits no values and completes immediately.
// Play: https://go.dev/play/p/D1JWkPG4NFK
func Empty[T any]() Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		destination.CompleteWithContext(ctx)

		return nil
	})
}

// Never creates an Observable that emits no values and never completes.
// This is useful for testing or when combining with other Observables.
// Play: https://go.dev/play/p/GHzcVYaEvN8
func Never() Observable[struct{}] {
	return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[struct{}]) Teardown {
		done := make(chan struct{})

		go func() {
			for {
				select {
				case <-subscriberCtx.Done():
					if subscriberCtx.Err() != nil {
						destination.ErrorWithContext(subscriberCtx, subscriberCtx.Err())
						return
					}

					destination.CompleteWithContext(subscriberCtx)
					return
				case <-done:
					return
				}
			}
		}()

		return func() {
			close(done)
		}
	})
}

// Throw creates an Observable that emits an error and completes immediately.
// Play: https://go.dev/play/p/1TBK8LdDRJF
func Throw[T any](err error) Observable[T] {
	// `nil` is a valid value for `err`
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		destination.ErrorWithContext(ctx, err)

		return nil
	})
}

// Defer creates an Observable that waits until an Observer subscribes to it,
// and then it creates an Observable for each Observer. This is useful for
// creating Observables that depend on some external state that is not
// available at the time of creation. The `cb` function is called for each
// Observer that subscribes to the Observable.
// Play: https://go.dev/play/p/wyVzordmkK0
func Defer[T any](factory func() Observable[T]) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		sub := factory().SubscribeWithContext(ctx, destination)

		return sub.Unsubscribe
	})
}

// Future creates an Observable that waits until an Observer subscribes to it,
// and then it emits either a value or an error, returned by the `factory` function.
//
// This is useful for creating Observables that depend on some external state
// that is not available at the time of creation. The `factory` function is called
// for each Observer that subscribes to the Observable.
func Future[T any](factory func() (T, error)) Observable[T] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[T]) Teardown {
		go func() {
			v, err := factory()
			if err != nil {
				destination.ErrorWithContext(ctx, err)
				return
			}

			destination.NextWithContext(ctx, v)
			destination.CompleteWithContext(ctx)
		}()

		return nil
	})
}

// Merge merges the values from all observables to a single observable result.
// It subscribes to each inner Observable, and emits all values
// from each inner Observable, maintaining their order. It completes when all
// inner Observables are done.
// Play: https://go.dev/play/p/hX2xPyeO3M9
func Merge[T any](sources ...Observable[T]) Observable[T] {
	return MergeAll[T]()(Just(sources...))
}

// CombineLatest2 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/mzpJyg7plnm
func CombineLatest2[A, B any](obsA Observable[A], obsB Observable[B]) Observable[lo.Tuple2[A, B]] {
	return CombineLatestWith1[A](obsB)(obsA)
}

// CombineLatest3 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
func CombineLatest3[A, B, C any](obsA Observable[A], obsB Observable[B], obsC Observable[C]) Observable[lo.Tuple3[A, B, C]] {
	return CombineLatestWith2[A](obsB, obsC)(obsA)
}

// CombineLatest4 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/mzpJyg7plnm
func CombineLatest4[A, B, C, D any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D]) Observable[lo.Tuple4[A, B, C, D]] {
	return CombineLatestWith3[A](obsB, obsC, obsD)(obsA)
}

// CombineLatest5 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/mzpJyg7plnm
func CombineLatest5[A, B, C, D, E any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E]) Observable[lo.Tuple5[A, B, C, D, E]] {
	return CombineLatestWith4[A](obsB, obsC, obsD, obsE)(obsA)
}

// CombineLatestAny combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/mzpJyg7plnm
func CombineLatestAny(sources ...Observable[any]) Observable[[]any] {
	return CombineLatestAllAny()(Just(sources...))
}

// Zip combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip[T any](sources ...Observable[T]) Observable[[]T] {
	return ZipAll[T]()(Just(sources...))
}

// Zip2 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip2[A, B any](obsA Observable[A], obsB Observable[B]) Observable[lo.Tuple2[A, B]] {
	return ZipWith1[A](obsB)(obsA)
}

// Zip3 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip3[A, B, C any](obsA Observable[A], obsB Observable[B], obsC Observable[C]) Observable[lo.Tuple3[A, B, C]] {
	return ZipWith2[A](obsB, obsC)(obsA)
}

// Zip4 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip4[A, B, C, D any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D]) Observable[lo.Tuple4[A, B, C, D]] {
	return ZipWith3[A](obsB, obsC, obsD)(obsA)
}

// Zip5 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip5[A, B, C, D, E any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E]) Observable[lo.Tuple5[A, B, C, D, E]] {
	return ZipWith4[A](obsB, obsC, obsD, obsE)(obsA)
}

// Zip6 combines the values from the source Observable with the latest
// values from the other Observables. It will only emit when all Observables have
// emitted at least one value. It completes when the source Observable completes.
// Play: https://go.dev/play/p/5YxbQ5jNzjQ
func Zip6[A, B, C, D, E, F any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E], obsF Observable[F]) Observable[lo.Tuple6[A, B, C, D, E, F]] {
	return ZipWith5[A](obsB, obsC, obsD, obsE, obsF)(obsA)
}

// Concat concatenates the source Observable with other Observables. It subscribes
// to each inner Observable only after the previous one completes, maintaining their
// order. It completes when all inner Observables are done.
// Play: https://go.dev/play/p/DFokqIXIguM
func Concat[T any](obs ...Observable[T]) Observable[T] {
	return ConcatAll[T]()(Just(obs...))
}

// Race creates an Observable that mirrors the first source Observable to
// emit a next, error or complete notification from the combination of the
// Observable sources. It cancels the subscriptions to all other Observables.
// It completes when the source Observable completes. If the source Observable
// emits an error, the error is emitted by the resulting Observable.
// Play: https://go.dev/play/p/5VzGFd62SMC
func Race[T any](sources ...Observable[T]) Observable[T] {
	if len(sources) == 0 {
		return Empty[T]()
	}

	return RaceWith(sources[1:]...)(sources[0])
}

// Amb is an alias for Race.
// Play: https://go.dev/play/p/-YvhnpQFVNS
func Amb[T any](sources ...Observable[T]) Observable[T] {
	return Race(sources...)
}

// RandIntN creates an Observable that emits random int values in the range [0, n).
// The count is the number of values to emit.
// Play: https://go.dev/play/p/4m7T5j-7i3a
func RandIntN(n, count int) Observable[int] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		for i := 0; i < count; i++ {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, xrand.IntN(n))
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

// RandFloat64 creates an Observable that emits random float64 values in the range [0, 1).
// The count is the number of values to emit.
// Play: https://go.dev/play/p/MRuy8rUpTve
func RandFloat64(count int) Observable[float64] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[float64]) Teardown {
		for i := 0; i < count; i++ {
			if destination.IsClosed() {
				return nil
			}

			destination.NextWithContext(ctx, xrand.Float64())
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}
