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
	"fmt"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorCombiningMergeWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	// parallel merge of synchronous sources: all 3 sources emit 3 values
	values, err := Collect(
		MergeWith[int64](
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
		)(
			Just[int64](0, 1, 2),
		),
	)
	is.Len(values, 9)
	is.NoError(err)
	is.ElementsMatch(values, []int64{0, 1, 2, 3, 4, 5, 6, 7, 8})

	// parallel: same values from 3 sources
	values, err = Collect(
		MergeWith[int64](
			Just[int64](0),
			Just[int64](0),
		)(
			Just[int64](0),
		),
	)
	is.Equal([]int64{0, 0, 0}, values)
	is.NoError(err)

	// empty source
	values, err = Collect(
		MergeWith[int64]()(
			Empty[int64](),
		),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	// error source
	values, err = Collect(
		MergeWith[int64]()(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith1(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// MergeWith is a parallel merge: all sources emit concurrently,
	// so use ElementsMatch to verify all values are present regardless of order.
	// Just(0,1,2) and RangeWithInterval(6,9,100ms): values 0,1,2 immediate, 6,7,8 delayed
	values, err := Collect(
		MergeWith1[int64](
			RangeWithInterval(6, 9, 100*time.Millisecond),
		)(
			Just[int64](0, 1, 2),
		),
	)
	is.Len(values, 6)
	is.ElementsMatch(values, []int64{0, 1, 2, 6, 7, 8})
	is.NoError(err)

	// parallel: two identical sources each emit 0,1,2 → 6 values total
	values, err = Collect(
		MergeWith1[int64](
			RangeWithInterval(0, 3, 100*time.Millisecond),
		)(
			RangeWithInterval(0, 3, 100*time.Millisecond),
		),
	)
	is.Len(values, 6)
	is.ElementsMatch(values, []int64{0, 0, 1, 1, 2, 2})
	is.NoError(err)

	// concurrent: Just(0) emits immediately, delayed source emits 0,1,2 after 100ms
	values, err = Collect(
		MergeWith1[int64](
			Delay[int64](100 * time.Millisecond)(RangeWithInterval(0, 3, 200*time.Millisecond)),
		)(
			Just[int64](0),
		),
	)
	is.Len(values, 4)
	is.ElementsMatch(values, []int64{0, 0, 1, 2})
	is.NoError(err)

	values, err = Collect(
		MergeWith1[int64](
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	// MergeWith subscribes concurrently; if Throw errors first,
	// Just(42) may not have emitted yet — only check error
	_, err = Collect(
		MergeWith1[int64](
			Throw[int64](assert.AnError),
		)(
			Just[int64](42),
		),
	)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith2(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith2[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
		)(
			Just[int64](6, 7, 8),
		),
	)
	is.Len(values, 9)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith2[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith3[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
		)(
			Just[int64](9, 10, 11),
		),
	)
	is.Len(values, 12)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith3[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith4[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
			Just[int64](9, 10, 11),
		)(
			Just[int64](12, 13, 14),
		),
	)
	is.Len(values, 15)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith4[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeWith5(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// parallel merge — order is non-deterministic
	values, err := Collect(
		MergeWith5[int64](
			Just[int64](0, 1, 2),
			Just[int64](3, 4, 5),
			Just[int64](6, 7, 8),
			Just[int64](9, 10, 11),
			Just[int64](12, 13, 14),
		)(
			Just[int64](15, 16, 17),
		),
	)
	is.Len(values, 18)
	is.ElementsMatch([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith5[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.NoError(err)

	values, err = Collect(
		MergeWith5[int64](
			Empty[int64](),
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Just[int64](42),
		),
	)
	is.Equal([]int64{42}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningMergeAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// sequential
	values, err := Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(6, 9, 100*time.Millisecond), // third
				RangeWithInterval(3, 6, 10*time.Millisecond),  // second
				Just[int64](0, 1, 2),                          // first
			),
		),
	)
	is.Equal([]int64{0, 1, 2, 3, 4, 5, 6, 7, 8}, values)
	is.NoError(err)

	// parallel
	values, err = Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(0, 3, 100*time.Millisecond),
				RangeWithInterval(0, 3, 100*time.Millisecond),
				RangeWithInterval(0, 3, 100*time.Millisecond),
			),
		),
	)
	is.Equal([]int64{0, 0, 0, 1, 1, 1, 2, 2, 2}, values)
	is.NoError(err)

	// concurrent
	values, err = Collect(
		MergeAll[int64]()(
			Just(
				RangeWithInterval(0, 3, 200*time.Millisecond),
				Delay[int64](66*time.Millisecond)(RangeWithInterval(3, 6, 200*time.Millisecond)),
				Delay[int64](132*time.Millisecond)(RangeWithInterval(6, 9, 200*time.Millisecond)),
			),
		),
	)
	is.Equal([]int64{0, 3, 6, 1, 4, 7, 2, 5, 8}, values)
	is.NoError(err)

	values, err = Collect(
		MergeAll[int64]()(Empty[Observable[int64]]()),
	)
	is.Equal([]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		MergeAll[int64]()(Throw[Observable[int64]](assert.AnError)),
	)
	is.Equal([]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	t.Run("inner lifecycle", func(t *testing.T) {
		testWithTimeout(t, 2000*time.Millisecond)
		is := assert.New(t)

		// teardowns counts inner teardowns; the inner never completes by itself.
		var teardowns int32
		pending := func() Observable[int] {
			return NewUnsafeObservable(func(Observer[int]) Teardown {
				return func() { atomic.AddInt32(&teardowns, 1) }
			})
		}

		// early unsubscription tears down active inners, and not the finished ones
		outer := NewPublishSubject[Observable[int]]()
		sub := MergeAll[int]()(outer).Subscribe(OnNext(func(int) {}))
		outer.Next(pending())
		outer.Next(Just(1)) // completes synchronously during subscription
		outer.Next(pending())
		is.Equal(int32(0), atomic.LoadInt32(&teardowns))
		sub.Unsubscribe()
		is.Equal(int32(2), atomic.LoadInt32(&teardowns))

		// an inner emitted after teardown is not subscribed to
		subscribed := false
		outer.Next(NewUnsafeObservable(func(Observer[int]) Teardown {
			subscribed = true
			return nil
		}))
		is.False(subscribed)

		// an inner error tears down sibling inners
		atomic.StoreInt32(&teardowns, 0)
		outer = NewPublishSubject[Observable[int]]()
		var gotErr error
		MergeAll[int]()(outer).Subscribe(NewObserver(
			func(int) {},
			func(err error) { gotErr = err },
			func() {},
		))
		outer.Next(pending())
		outer.Next(Throw[int](assert.AnError))
		is.EqualError(gotErr, assert.AnError.Error())
		is.Equal(int32(1), atomic.LoadInt32(&teardowns))
	})
}

func TestOperatorCombiningMergeAll_releasesCompletedInners(t *testing.T) { //nolint:paralleltest
	// Not parallel: it measures the process-wide heap.
	is := assert.New(t)

	// Retaining one completed inner costs hundreds of bytes, so the old
	// behavior grows the heap by tens of MB for this many inners.
	const (
		innerCount   = 100_000
		maxHeapGrowB = 8 << 20
	)

	heapAlloc := func() uint64 {
		runtime.GC()
		runtime.GC()

		var m runtime.MemStats
		runtime.ReadMemStats(&m)

		return m.HeapAlloc
	}

	outer := NewPublishSubject[Observable[int]]()
	received := int64(0)
	completed := false

	sub := MergeAll[int]()(outer).Subscribe(NewObserver(
		func(int) { received++ },
		func(error) {},
		func() { completed = true },
	))

	before := heapAlloc()

	for i := 0; i < innerCount; i++ {
		outer.Next(Just(i))
	}

	after := heapAlloc()

	is.Equal(int64(innerCount), received)
	is.Less(int64(after)-int64(before), int64(maxHeapGrowB))

	outer.Complete()
	sub.Wait()
	is.True(completed)
}

func TestOperatorCombiningMergeMap(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			RangeWithInterval(3, 7, 150*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{"3", "3", "3", "4", "4", "4", "4", "5", "5", "5", "5", "5", "6", "6", "6", "6", "6", "6"}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(0, 2, 50*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), 3, 100*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{"0", "1", "0", "1", "0", "1"}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int64](),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			RangeWithInterval(3, 7, 30*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return Empty[string]()
			}),
		),
	)
	is.Equal([]string{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int64](assert.AnError),
			MergeMap(func(item int64) Observable[string] {
				return RepeatWithInterval(strconv.Itoa(int(item)), item, 20*time.Millisecond)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			RangeWithInterval(3, 7, 30*time.Millisecond),
			MergeMap(func(item int64) Observable[string] {
				return Throw[string](assert.AnError)
			}),
		),
	)
	is.Equal([]string{}, values)
	is.EqualError(err, assert.AnError.Error())

	t.Run("MergeMapI index restarts on each subscription", func(t *testing.T) {
		is := assert.New(t)

		obs := MergeMapI(func(item string, index int64) Observable[int64] {
			return Just(index)
		})(Just("a", "b", "c"))

		// Each subscription must restart its index at 0.
		for i := 0; i < 2; i++ {
			values, err := Collect(obs)
			is.Equal([]int64{0, 1, 2}, values)
			is.NoError(err)
		}
	})
}

func TestOperatorCombiningCombineLatestWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// check type is in the right order
	values2, err := Collect(
		CombineLatestWith[int](
			Of("42"),
		)(
			Of(42),
		),
	)
	is.Equal([]lo.Tuple2[int, string]{lo.T2(42, "42")}, values2)
	is.NoError(err)

	values1, err := Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 2, 50*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 75*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(0)), lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 2, 20*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			RangeWithInterval(0, 3, 10*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(2)), lo.T2(int64(1), int64(2))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith[int64](
			Throw[int64](assert.AnError),
		)(
			Delay[int64](10 * time.Millisecond)(Of[int64](42)),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())

	values1, err = Collect(
		CombineLatestWith[int64](
			Delay[int64](10 * time.Millisecond)(Throw[int64](assert.AnError)),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith1(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// check type is in the right order
	values2, err := Collect(
		CombineLatestWith1[int](
			Of("42"),
		)(
			Of(42),
		),
	)
	is.Equal([]lo.Tuple2[int, string]{lo.T2(42, "42")}, values2)
	is.NoError(err)

	values1, err := Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 2, 50*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 75*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(0)), lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 2, 20*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(1))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			RangeWithInterval(0, 3, 10*time.Millisecond),
		)(
			RangeWithInterval(0, 2, 100*time.Millisecond),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(2)), lo.T2(int64(1), int64(2))}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.NoError(err)

	values1, err = Collect(
		CombineLatestWith1[int64](
			Throw[int64](assert.AnError),
		)(
			Delay[int64](10 * time.Millisecond)(Of[int64](42)),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())

	values1, err = Collect(
		CombineLatestWith1[int64](
			Delay[int64](10 * time.Millisecond)(Throw[int64](assert.AnError)),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values1)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith2(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// CombineLatest emits a tuple whenever any source emits after all sources have emitted at least once
	// With Of (single-value) sources, only one tuple is produced
	values, err := Collect(
		CombineLatestWith2[int64](
			Of[int64](1),
			Of[int64](2),
		)(
			Of[int64](3),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC) — source is the function argument, not the variadic params
	is.Equal([]lo.Tuple3[int64, int64, int64]{lo.T3(int64(3), int64(1), int64(2))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith2[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestWith3[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
		)(
			Of[int64](4),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC, argD) — source is the function argument
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{lo.T4(int64(4), int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith3[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Of[int64](5),
		),
	)
	is.Len(values, 1)
	// Tuple is (source, argB, argC, argD, argE) — source is the function argument
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{lo.T5(int64(5), int64(1), int64(2), int64(3), int64(4))}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestWith4[int64](
			Throw[int64](assert.AnError),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Of[int64](42),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](21),
				Of[int64](42),
			),
		),
	)
	is.Equal([][]int64{{21, 42}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 150*time.Millisecond),
				RangeWithInterval(0, 2, 100*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 0}, {0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 2, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 1}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 3, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 2}, {1, 2}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](42),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Empty[int64](),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Delay[int64](10*time.Millisecond)(Of[int64](42)),
				Throw[int64](assert.AnError),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		CombineLatestAll[int64]()(
			Just(
				Of[int64](42),
				Delay[int64](10*time.Millisecond)(Throw[int64](assert.AnError)),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningCombineLatestAllAny(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](21),
				Of[any]("42"),
			),
		),
	)
	is.Equal([][]any{{21, "42"}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 150*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(0)}, {int64(0), int64(1)}, {int64(1), int64(1)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 20*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(1)}, {int64(1), int64(1)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Map(func(x int64) any { return x })(RangeWithInterval(0, 2, 100*time.Millisecond)),
				Map(func(x int64) any { return x })(RangeWithInterval(0, 3, 10*time.Millisecond)),
			),
		),
	)
	is.Equal([][]any{{int64(0), int64(2)}, {int64(1), int64(2)}}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](42),
				Empty[any](),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Empty[any](),
				Empty[any](),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.NoError(err)

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Delay[any](10*time.Millisecond)(Of[any](42)),
				Throw[any](assert.AnError),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		CombineLatestAllAny()(
			Just(
				Of[any](42),
				Delay[any](10*time.Millisecond)(Throw[any](assert.AnError)),
			),
		),
	)
	is.Equal([][]any{}, values)
	is.EqualError(err, assert.AnError.Error())
}

// A source completing without any value makes a tuple impossible: the result
// must complete at once instead of waiting for the other sources.
func TestOperatorCombiningCombineLatest_emptyAndNever(t *testing.T) { //nolint:paralleltest
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	values1, err := Collect(CombineLatest2(Empty[int](), Never()))
	is.Equal([]lo.Tuple2[int, struct{}]{}, values1)
	is.NoError(err)

	values1b, err := Collect(CombineLatest2(Never(), Empty[int]()))
	is.Equal([]lo.Tuple2[struct{}, int]{}, values1b)
	is.NoError(err)

	values3, err := Collect(CombineLatest3(Never(), Empty[int](), Never()))
	is.Equal([]lo.Tuple3[struct{}, int, struct{}]{}, values3)
	is.NoError(err)

	values4, err := Collect(CombineLatest4(Never(), Never(), Never(), Empty[int]()))
	is.Equal([]lo.Tuple4[struct{}, struct{}, struct{}, int]{}, values4)
	is.NoError(err)

	values5, err := Collect(CombineLatest5(Never(), Never(), Never(), Never(), Empty[int]()))
	is.Equal([]lo.Tuple5[struct{}, struct{}, struct{}, struct{}, int]{}, values5)
	is.NoError(err)

	neverAny := NewObservable(func(Observer[any]) Teardown { return nil })

	valuesAll, err := Collect(CombineLatestAny(Empty[any](), neverAny))
	is.Equal([][]any{}, valuesAll)
	is.NoError(err)

	valuesAll, err = Collect(CombineLatestAny(neverAny, Empty[any]()))
	is.Equal([][]any{}, valuesAll)
	is.NoError(err)
}

func TestOperatorCombiningConcatWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	// sequential concatenation
	values, err := Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	// concatenate multiple observables
	values, err = Collect(
		ConcatWith(
			Just(4, 5),
			Just(6, 7),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6, 7}, values)
	is.NoError(err)

	// empty source
	values, err = Collect(
		ConcatWith(
			Just(1, 2, 3),
		)(
			Empty[int](),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// empty additional observable
	values, err = Collect(
		ConcatWith(
			Empty[int](),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// error propagation from source: source Throw is the first inner observable,
	// so ConcatAll errors immediately without emitting values.
	values, err = Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Throw[int](assert.AnError),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	// error propagation from additional observable
	values, err = Collect(
		ConcatWith(
			Just(4, 5, 6),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	// error from additional observable after source completes
	values, err = Collect(
		ConcatWith(
			Throw[int](assert.AnError),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningConcatAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ConcatAll[int]()(
			Just(
				Just(1, 2, 3),
				Just(4, 5, 6),
			),
		),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		ConcatAll[int]()(Empty[Observable[int]]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		ConcatAll[int]()(Throw[Observable[int]](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningStartWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		StartWith(1, 2, 3)(Just(4, 5, 6)),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith(1, 2, 3)(Empty[int]()),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		StartWith(1, 2, 3)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningEndWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		EndWith(1, 2, 3)(Just(4, 5, 6)),
	)
	is.Equal([]int{4, 5, 6, 1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith[int]()(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith(1, 2, 3)(Empty[int]()),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		EndWith(1, 2, 3)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningPairwise(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pairwise[int]()(Of(0)),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2}, {2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Empty[int]()),
	)
	is.Equal([][]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pairwise[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([][]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningRaceWith(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1500*time.Millisecond)
	is := assert.New(t)

	// empty
	values, err := Collect(
		RaceWith[int]()(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// async
	values, err = Collect(
		RaceWith(
			Delay[int](150*time.Millisecond)(Just(4, 5, 6)),
			Delay[int](50*time.Millisecond)(Just(7, 8, 9)),
			Delay[int](200*time.Millisecond)(Just(10, 11, 12)),
		)(
			Delay[int](100 * time.Millisecond)(Just(1, 2, 3)),
		),
	)
	is.Equal([]int{7, 8, 9}, values)
	is.NoError(err)

	// sequential
	values, err = Collect(
		RaceWith(
			Just(4, 5, 6),
			Just(7, 8, 9),
			Just(10, 11, 12),
		)(
			Just(1, 2, 3),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	// mixed async + sequential
	values, err = Collect(
		RaceWith(
			Delay[int](150*time.Millisecond)(Just(4, 5, 6)),
			Just(4, 5, 6),
			Delay[int](200*time.Millisecond)(Just(10, 11, 12)),
		)(
			Delay[int](100 * time.Millisecond)(Just(1, 2, 3)),
		),
	)
	is.Equal([]int{4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Race(Empty[int](), Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Race[int](),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Race(
			Delay[int](250*time.Millisecond)(Just(1, 2, 3)),
			Delay[int](25*time.Millisecond)(Throw[int](assert.AnError)),
			Delay[int](250*time.Millisecond)(Just(7, 8, 9)),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 10),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith[int64](
			Of[int64](42),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith1(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith1[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Skip[int64](1)(Range(0, 4)),
		)(
			Range(0, 10),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{lo.T2(int64(0), int64(1)), lo.T2(int64(1), int64(2)), lo.T2(int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith1[int64](
			Of[int64](42),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple2[int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith2(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith2[int64](
			Skip[int64](1)(Range(0, 4)),
			Skip[int64](2)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{lo.T3(int64(0), int64(1), int64(2)), lo.T3(int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Range(0, 4),
			Range(0, 4),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith2[int64](
			Of[int64](42),
			Of[int64](99),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple3[int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith3(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith3[int64](
			Skip[int64](1)(Range(0, 4)),
			Skip[int64](2)(Range(0, 4)),
			Skip[int64](3)(Range(0, 4)),
		)(
			Range(0, 4),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{lo.T4(int64(0), int64(1), int64(2), int64(3))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith3[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith3[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple4[int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith4(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Of[int64](5),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{lo.T5(int64(5), int64(1), int64(2), int64(3), int64(4))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith4[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith4[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple5[int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipWith5(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	// ZipWith5 is strict — all sources must have values; tuple is (source, arg1, arg2, arg3, arg4, arg5)
	values, err := Collect(
		ZipWith5[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
			Of[int64](5),
		)(
			Of[int64](0),
		),
	)
	is.Len(values, 1)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{lo.T6(int64(0), int64(1), int64(2), int64(3), int64(4), int64(5))}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith5[int64](
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
			Empty[int64](),
		)(
			Empty[int64](),
		),
	)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipWith5[int64](
			Of[int64](1),
			Of[int64](2),
			Of[int64](3),
			Of[int64](4),
			Of[int64](5),
		)(
			Throw[int64](assert.AnError),
		),
	)
	is.Equal([]lo.Tuple6[int64, int64, int64, int64, int64, int64]{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorCombiningZipAll(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 2000*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](21),
				Of[int64](42),
			),
		),
	)
	is.Equal([][]int64{{21, 42}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 150*time.Millisecond),
				RangeWithInterval(0, 2, 100*time.Millisecond),
			),
		),
	)
	// Zip pairs positionally (1st with 1st, 2nd with 2nd), buffering the faster
	// source's values until the slower one catches up. It never reuses a value
	// already paired, regardless of which source is faster.
	is.Equal([][]int64{{0, 0}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 2, 20*time.Millisecond),
			),
		),
	)
	is.Equal([][]int64{{0, 0}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				RangeWithInterval(0, 2, 200*time.Millisecond),
				RangeWithInterval(0, 3, 20*time.Millisecond),
			),
		),
	)
	// The 2-item source completes after 2 pairs: the faster source's 3rd
	// buffered value is discarded, since it can never be paired.
	is.Equal([][]int64{{0, 0}, {1, 1}}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](42),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Empty[int64](),
				Empty[int64](),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.NoError(err)

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Delay[int64](10*time.Millisecond)(Of[int64](42)),
				Throw[int64](assert.AnError),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		ZipAll[int64]()(
			Just(
				Of[int64](42),
				Delay[int64](10*time.Millisecond)(Throw[int64](assert.AnError)),
			),
		),
	)
	is.Equal([][]int64{}, values)
	is.EqualError(err, assert.AnError.Error())
}

// zipVariant wraps one zip operator behind a common shape, so every test runs against all of them.
type zipVariant struct {
	name  string
	arity int
	// zip emits each group of paired values as a slice, whatever the operator's output type.
	zip func(sources []Observable[int]) Observable[[]int]
}

func zipCompletionVariants() []zipVariant {
	return []zipVariant{
		{"Zip", 2, func(s []Observable[int]) Observable[[]int] { return Zip(s...) }},
		{"ZipAll", 2, func(s []Observable[int]) Observable[[]int] { return ZipAll[int]()(Just(s...)) }},
		{"Zip2", 2, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple2[int, int]) []int { return []int{v.A, v.B} })(Zip2(s[0], s[1]))
		}},
		{"Zip3", 3, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple3[int, int, int]) []int { return []int{v.A, v.B, v.C} })(Zip3(s[0], s[1], s[2]))
		}},
		{"Zip4", 4, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple4[int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D} })(Zip4(s[0], s[1], s[2], s[3]))
		}},
		{"Zip5", 5, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple5[int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E} })(Zip5(s[0], s[1], s[2], s[3], s[4]))
		}},
		{"Zip6", 6, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple6[int, int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E, v.F} })(Zip6(s[0], s[1], s[2], s[3], s[4], s[5]))
		}},
	}
}

// manualSources are sources that emit nothing by themselves: tests push values through emitters.
type manualSources struct {
	observables []Observable[int]
	// emitters[i] is set once source i is subscribed.
	emitters []Observer[int]
}

// newManualSources builds arity sources. onSubscribe (optional) runs when source i is subscribed,
// and onTeardown (optional) runs when its teardown is called.
func newManualSources(
	arity int,
	onSubscribe func(ctx context.Context, i int, destination Observer[int]),
	onTeardown func(i int),
) *manualSources {
	m := &manualSources{
		observables: make([]Observable[int], arity),
		emitters:    make([]Observer[int], arity),
	}
	for i := range m.observables {
		i := i
		m.observables[i] = NewObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
			m.emitters[i] = destination
			if onSubscribe != nil {
				onSubscribe(ctx, i, destination)
			}
			return func() {
				if onTeardown != nil {
					onTeardown(i)
				}
			}
		})
	}
	return m
}

// zipRow is the group zip emits for the given round when source i emitted round*10+i.
func zipRow(round, arity int) []int {
	row := make([]int, arity)
	for i := range row {
		row[i] = round*10 + i
	}
	return row
}

func TestOperatorCombiningZip_completedSource(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		for short := 0; short < variant.arity; short++ {
			short := short
			t.Run(fmt.Sprintf("%s/source%d", variant.name, short), func(t *testing.T) {
				t.Parallel()
				testWithTimeout(t, 5*time.Second)
				is := assert.New(t)

				type contextKey struct{}
				subscriberCtx := context.WithValue(context.Background(), contextKey{}, "subscriber")
				emissionCtx := context.WithValue(subscriberCtx, contextKey{}, "last value")

				teardowns := make([]int, variant.arity)
				sources := newManualSources(variant.arity,
					func(ctx context.Context, i int, destination Observer[int]) {
						is.Equal(subscriberCtx, ctx)
						// Source `short` emits two values, then completes, as soon as it is subscribed.
						if i == short {
							destination.NextWithContext(ctx, 10+i)
							destination.NextWithContext(ctx, 20+i)
							destination.CompleteWithContext(ctx)
						}
					},
					func(i int) { teardowns[i]++ },
				)

				values := [][]int{}
				completions := 0
				sub := variant.zip(sources.observables).SubscribeWithContext(subscriberCtx, NewObserverWithContext(
					func(ctx context.Context, value []int) {
						is.Equal(emissionCtx, ctx)
						values = append(values, value)
					},
					func(_ context.Context, err error) { is.NoError(err) },
					func(ctx context.Context) {
						is.Equal(emissionCtx, ctx)
						completions++
					},
				))
				defer sub.Unsubscribe()

				// Subscribe has returned, so completion now runs Zip's teardown.
				// The completed source must retain its second value until it is paired.
				for round := 1; round <= 2; round++ {
					for i, emitter := range sources.emitters {
						if i != short {
							emitter.NextWithContext(emissionCtx, round*10+i)
						}
					}
					is.Len(values, round)
					is.Equal(round-1, completions)
				}

				is.Equal([][]int{zipRow(1, variant.arity), zipRow(2, variant.arity)}, values)
				is.True(sub.IsClosed())
				for _, count := range teardowns {
					is.Equal(1, count)
				}
			})
		}
	}
}

func TestOperatorCombiningZip_futureCompletion(t *testing.T) {
	t.Parallel()

	// A source completing while another goroutine delivers the last pair must not drop it.
	// The window is a few instructions wide, so repeat to hit it reliably.
	const iterations = 10

	for _, variant := range zipCompletionVariants() {
		variant := variant
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()
			testWithTimeout(t, 5*time.Second)
			is := assert.New(t)

			want := make([]int, variant.arity)
			for i := range want {
				want[i] = i
			}

			for n := 0; n < iterations; n++ {
				release := make(chan struct{})
				sources := make([]Observable[int], variant.arity)
				for i := range sources {
					i := i
					sources[i] = Future(func() (int, error) {
						<-release
						return i, nil
					})
				}

				// Collect blocks until completion, so release the futures concurrently.
				go close(release)
				values, err := Collect(variant.zip(sources))
				is.NoError(err)
				if !is.Equal([][]int{want}, values) {
					return
				}
			}
		})
	}
}

func TestOperatorCombiningZip_unsubscribeFromNext(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()
			testWithTimeout(t, 5*time.Second)
			is := assert.New(t)

			teardowns := 0
			sources := newManualSources(variant.arity, nil, func(int) { teardowns++ })

			values := 0
			completions := 0
			var sub Subscription
			sub = variant.zip(sources.observables).Subscribe(NewObserver(
				func(_ []int) {
					values++
					sub.Unsubscribe()
				},
				func(err error) { is.NoError(err) },
				func() { completions++ },
			))
			defer sub.Unsubscribe()

			for _, emitter := range sources.emitters {
				emitter.Next(1)
			}

			is.Equal(1, values)
			is.Zero(completions)
			is.True(sub.IsClosed())
			is.Equal(variant.arity, teardowns)
		})
	}
}

func TestOperatorCombiningZip_terminalCleanup(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		for _, terminal := range []string{"complete", "error", "cancel"} {
			terminal := terminal
			t.Run(variant.name+"/"+terminal, func(t *testing.T) {
				t.Parallel()
				testWithTimeout(t, 5*time.Second)
				is := assert.New(t)

				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()

				// Teardown may run asynchronously (on cancel), so signal it through a channel.
				teardowns := make(chan struct{}, variant.arity)
				sources := newManualSources(variant.arity, nil, func(int) { teardowns <- struct{}{} })
				observables := sources.observables
				if terminal == "cancel" {
					last := len(observables) - 1
					observables[last] = ThrowOnContextCancel[int]()(observables[last])
				}

				values := 0
				completions := 0
				var gotErr error
				sub := variant.zip(observables).SubscribeWithContext(ctx, NewObserver(
					func(_ []int) { values++ },
					func(err error) { gotErr = err },
					func() { completions++ },
				))
				defer sub.Unsubscribe()

				// Any source can end the stream: use the last one subscribed.
				last := sources.emitters[len(sources.emitters)-1]
				switch terminal {
				case "complete":
					last.Complete()
				case "error":
					last.Error(assert.AnError)
				case "cancel":
					cancel()
				}
				for range observables {
					<-teardowns
				}

				is.Zero(values)
				is.True(sub.IsClosed())
				switch terminal {
				case "complete":
					is.Equal(1, completions)
					is.NoError(gotErr)
				case "error":
					is.Zero(completions)
					is.ErrorIs(gotErr, assert.AnError)
				case "cancel":
					is.Zero(completions)
					is.ErrorIs(gotErr, context.Canceled)
				}
			})
		}
	}
}

// A synchronous first source that fails closes the destination: the remaining sources must not be subscribed.
func TestOperatorCombining_skipSubscriptionWhenDestinationClosed(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name  string
		build func(late Observable[int]) Observable[int]
	}{
		{"CombineLatestWith1", func(late Observable[int]) Observable[int] {
			return Pipe2(Throw[int](assert.AnError), CombineLatestWith1[int](late), Map(func(lo.Tuple2[int, int]) int { return 0 }))
		}},
		{"CombineLatestWith4", func(late Observable[int]) Observable[int] {
			return Pipe2(Throw[int](assert.AnError), CombineLatestWith4[int](late, late, late, late), Map(func(lo.Tuple5[int, int, int, int, int]) int { return 0 }))
		}},
		{"CombineLatestAll", func(late Observable[int]) Observable[int] {
			return Pipe2(Just(Throw[int](assert.AnError), late, late), CombineLatestAll[int](), Map(func([]int) int { return 0 }))
		}},
		{"ZipWith1", func(late Observable[int]) Observable[int] {
			return Pipe2(Throw[int](assert.AnError), ZipWith1[int](late), Map(func(lo.Tuple2[int, int]) int { return 0 }))
		}},
		{"ZipWith5", func(late Observable[int]) Observable[int] {
			return Pipe2(Throw[int](assert.AnError), ZipWith5[int](late, late, late, late, late), Map(func(lo.Tuple6[int, int, int, int, int, int]) int { return 0 }))
		}},
		{"ZipAll", func(late Observable[int]) Observable[int] {
			return Pipe2(Just(Throw[int](assert.AnError), late, late), ZipAll[int](), Map(func([]int) int { return 0 }))
		}},
	}

	for _, tt := range tests {
		tt := tt

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			var subscribed int32

			late := Defer(func() Observable[int] {
				atomic.AddInt32(&subscribed, 1)
				return Just(1)
			})

			_, err := Collect(tt.build(late))
			is.EqualError(err, assert.AnError.Error())
			is.Equal(int32(0), atomic.LoadInt32(&subscribed))
		})
	}
}
