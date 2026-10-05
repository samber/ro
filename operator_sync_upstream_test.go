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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// countingSyncSource emits 0..total-1 synchronously, stops when its destination is
// closed, and counts the items it tried to emit.
func countingSyncSource(total int, emitted *int64) Observable[int] {
	return NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		for i := 0; i < total; i++ {
			if destination.IsClosed() {
				return nil
			}

			atomic.AddInt64(emitted, 1)
			destination.NextWithContext(ctx, i)
		}

		destination.CompleteWithContext(ctx)

		return nil
	})
}

func TestOperatorSyncUpstreamStopsAfterResult(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)

	const total = 1000

	tests := []struct {
		name            string
		operator        func(Observable[int]) Observable[int]
		expectedValues  []int
		expectedErr     error
		expectedEmitted int64
	}{
		{
			name:            "Take",
			operator:        Take[int](2),
			expectedValues:  []int{0, 1},
			expectedEmitted: 2,
		},
		{
			name:            "Take one",
			operator:        Take[int](1),
			expectedValues:  []int{0},
			expectedEmitted: 1,
		},
		{
			name:            "Head",
			operator:        Head[int](),
			expectedValues:  []int{0},
			expectedEmitted: 1,
		},
		{
			name:            "First",
			operator:        First(func(v int) bool { return v == 3 }),
			expectedValues:  []int{3},
			expectedEmitted: 4,
		},
		{
			name:            "ElementAt",
			operator:        ElementAt[int](3),
			expectedValues:  []int{3},
			expectedEmitted: 4,
		},
		{
			name:            "ElementAt first",
			operator:        ElementAt[int](0),
			expectedValues:  []int{0},
			expectedEmitted: 1,
		},
		{
			name:            "ElementAtOrDefault",
			operator:        ElementAtOrDefault(3, -1),
			expectedValues:  []int{3},
			expectedEmitted: 4,
		},
		{
			name:            "TakeWhile",
			operator:        TakeWhile(func(v int) bool { return v < 3 }),
			expectedValues:  []int{0, 1, 2},
			expectedEmitted: 4,
		},
		{
			name:            "TakeWhile first item rejected",
			operator:        TakeWhile(func(v int) bool { return false }),
			expectedValues:  []int{},
			expectedEmitted: 1,
		},
		{
			name: "MapErr",
			operator: MapErr(func(v int) (int, error) {
				if v == 3 {
					return 0, assert.AnError
				}

				return v, nil
			}),
			expectedValues:  []int{0, 1, 2},
			expectedErr:     assert.AnError,
			expectedEmitted: 4,
		},
	}

	for _, tt := range tests {
		var emitted int64

		values, err := Collect(tt.operator(countingSyncSource(total, &emitted)))

		assert.Equal(t, tt.expectedValues, values, tt.name)

		if tt.expectedErr != nil {
			assert.EqualError(t, err, tt.expectedErr.Error(), tt.name)
		} else {
			assert.NoError(t, err, tt.name)
		}

		assert.Equal(t, tt.expectedEmitted, atomic.LoadInt64(&emitted), tt.name)
	}
}

// An unbounded synchronous source never returns unless the operator stops it.
func TestOperatorSyncUpstreamStopsInfiniteSource(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(Take[int64](2)(Range(0, math.MaxInt64)))
	is.Equal([]int64{0, 1}, values)
	is.NoError(err)

	values, err = Collect(Head[int64]()(Repeat(int64(7), math.MaxInt64)))
	is.Equal([]int64{7}, values)
	is.NoError(err)

	values, err = Collect(ElementAt[int64](2)(Range(0, math.MaxInt64)))
	is.Equal([]int64{2}, values)
	is.NoError(err)

	values, err = Collect(TakeWhile(func(v int64) bool { return v < 2 })(Range(0, math.MaxInt64)))
	is.Equal([]int64{0, 1}, values)
	is.NoError(err)

	values, err = Collect(First(func(v int64) bool { return v == 2 })(Range(0, math.MaxInt64)))
	is.Equal([]int64{2}, values)
	is.NoError(err)

	floats, err := Collect(Take[float64](2)(RangeWithStep(0, math.MaxInt32, 1)))
	is.Equal([]float64{0, 1}, floats)
	is.NoError(err)

	values2, err := Collect(Take[int](2)(FromSlice(make([]int, 1000), make([]int, 1000))))
	is.Equal([]int{0, 0}, values2)
	is.NoError(err)
}

// Stopping the upstream must not break an asynchronous source.
func TestOperatorSyncUpstreamStopsAsyncSource(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	var tornDown int32

	source := NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		stop := make(chan struct{})

		go func() {
			for i := 0; ; i++ {
				select {
				case <-stop:
					return
				case <-time.After(time.Millisecond):
					destination.NextWithContext(ctx, i)
				}
			}
		}()

		return func() {
			atomic.AddInt32(&tornDown, 1)
			close(stop)
		}
	})

	values, err := Collect(Take[int](3)(source))
	is.Equal([]int{0, 1, 2}, values)
	is.NoError(err)

	is.Eventually(func() bool { return atomic.LoadInt32(&tornDown) == 1 }, time.Second, time.Millisecond)
}

func TestOperatorCombiningConcatDoesNotBlockSubscribe(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	released := make(chan struct{})
	first := NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		go func() {
			<-released
			destination.NextWithContext(ctx, 1)
			destination.CompleteWithContext(ctx)
		}()

		return nil
	})

	var got []int

	completed := make(chan struct{})

	subscribed := make(chan Subscription, 1)
	go func() {
		subscribed <- Concat(first, Just(2, 3)).Subscribe(NewObserver(
			func(v int) { got = append(got, v) },
			func(err error) {},
			func() { close(completed) },
		))
	}()

	var sub Subscription

	select {
	case sub = <-subscribed:
	case <-time.After(200 * time.Millisecond):
		t.Fatal("Subscribe blocked while the first inner Observable is still active")
	}

	close(released)
	<-completed
	sub.Wait()

	is.Equal([]int{1, 2, 3}, got)
}

func TestOperatorCombiningConcatUnsubscribeStopsActiveInner(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	var tornDown, secondSubscribed int32

	first := NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		return func() { atomic.AddInt32(&tornDown, 1) }
	})
	second := NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		atomic.AddInt32(&secondSubscribed, 1)
		destination.CompleteWithContext(ctx)

		return nil
	})

	sub := Concat(first, second).Subscribe(NoopObserver[int]())
	sub.Unsubscribe()

	is.Equal(int32(1), atomic.LoadInt32(&tornDown))
	is.Equal(int32(0), atomic.LoadInt32(&secondSubscribed))
}

func TestOperatorCombiningConcatErrorSkipsNextSources(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	var subscribed int32

	next := NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		atomic.AddInt32(&subscribed, 1)
		destination.NextWithContext(ctx, 99)
		destination.CompleteWithContext(ctx)

		return nil
	})

	values, err := Collect(Concat(Just(1), Throw[int](assert.AnError), next, next))
	is.Equal([]int{1}, values)
	is.EqualError(err, assert.AnError.Error())
	is.Equal(int32(0), atomic.LoadInt32(&subscribed))

	// outer error
	values, err = Collect(ConcatAll[int]()(Throw[Observable[int]](assert.AnError)))
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningConcatAsyncInnerKeepsOrder(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	async := func(v int) Observable[int] {
		return NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			go func() {
				time.Sleep(5 * time.Millisecond)
				destination.NextWithContext(ctx, v)
				destination.CompleteWithContext(ctx)
			}()

			return nil
		})
	}

	values, err := Collect(Concat(Just(0), async(1), Empty[int](), async(2), Just(3)))
	is.Equal([]int{0, 1, 2, 3}, values)
	is.NoError(err)

	// inner error after an async inner
	values, err = Collect(Concat(async(1), Throw[int](assert.AnError), async(2)))
	is.Equal([]int{1}, values)
	is.EqualError(err, assert.AnError.Error())
}

// Synchronous inner Observables must not grow the stack with their number.
func TestOperatorCombiningConcatManySyncInners(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 2*time.Second)
	is := assert.New(t)

	const count = 100_000

	inners := make([]Observable[int], count)
	for i := range inners {
		inners[i] = Just(1)
	}

	values, err := Collect(Concat(inners...))
	is.Len(values, count)
	is.NoError(err)
}
