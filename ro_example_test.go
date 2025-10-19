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
	"errors"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func ExampleNewObserver() {
	observer := NewObserver(
		func(value int) {
			fmt.Printf("Next: %d\n", value)
		},
		func(err error) {
			fmt.Printf("Error: %s\n", err.Error())
		},
		func() {
			fmt.Printf("Completed\n")
		},
	)

	observer.Next(123)  // 123 logged
	observer.Next(456)  // 456 logged
	observer.Complete() // Completed logged

	observer.Next(789) // nothing logged

	// Output:
	// Next: 123
	// Next: 456
	// Completed
}

func ExampleNewObserver_error() {
	observer := NewObserver(
		func(value int) {
			fmt.Printf("Next: %d\n", value)
		},
		func(err error) {
			fmt.Printf("Error: %s\n", err.Error())
		},
		func() {
			fmt.Printf("Completed\n")
		},
	)

	observer.Next(123)             // 123 logged
	observer.Next(456)             // 456 logged
	observer.Error(assert.AnError) // Completed logged

	observer.Next(789) // nothing logged

	// Output:
	// Next: 123
	// Next: 456
	// Error: assert.AnError general error for testing
}

func ExampleNewObserver_empty() {
	observer := NewObserver(
		func(value int) {
			fmt.Printf("Next: %d\n", value)
		},
		func(err error) {
			fmt.Printf("Error: %s\n", err.Error())
		},
		func() {
			fmt.Printf("Completed\n")
		},
	)

	observer.Complete() // Completed logged

	observer.Next(123) // nothing logged

	// Output:
	// Completed
}

func ExampleMergeWith_ok() {
	observable := Pipe1(
		Just(1, 2),
		MergeWith(Delay[int](20*time.Millisecond)(Just(3, 4))),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleMergeWith_error() {
	observable := Pipe1(
		Throw[int](assert.AnError),
		MergeWith(Just(1, 2, 3, 4)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleMergeWith1_ok() {
	observable := Pipe1(
		Delay[int](20*time.Millisecond)(Just(2)),
		MergeWith(Just(1)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(30 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleMergeWith1_error() {
	observable := Pipe1(
		Delay[int](20*time.Millisecond)(Throw[int](assert.AnError)),
		MergeWith(Just(1)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(30 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Error: assert.AnError general error for testing
}

func ExampleMergeWith2_ok() {
	observable := Pipe1(
		Delay[int](50*time.Millisecond)(Just(4, 5)),
		MergeWith2(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3)),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleMergeWith2_error() {
	observable := Pipe1(
		Delay[int](50*time.Millisecond)(Throw[int](assert.AnError)),
		MergeWith2(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3)),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleMergeWith3_ok() {
	observable := Pipe1(
		Delay[int](75*time.Millisecond)(Just(7, 8)),
		MergeWith3(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(100 * time.Millisecond)
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 7
	// Next: 8
	// Completed
}

func ExampleMergeWith3_error() {
	observable := Pipe1(
		Delay[int](75*time.Millisecond)(Throw[int](assert.AnError)),
		MergeWith3(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
		),
	)
	subscription := observable.Subscribe(PrintObserver[int]())
	time.Sleep(100 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Error: assert.AnError general error for testing
}

func ExampleMergeWith4_ok() {
	observable := Pipe1(
		Delay[int](100*time.Millisecond)(Just(9, 10)),
		MergeWith4(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
			Delay[int](75*time.Millisecond)(Just(7, 8)),
		),
	)
	subscription := observable.Subscribe(PrintObserver[int]())
	time.Sleep(120 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 7
	// Next: 8
	// Next: 9
	// Next: 10
	// Completed
}

func ExampleMergeWith4_error() {
	observable := Pipe1(
		Delay[int](100*time.Millisecond)(Throw[int](assert.AnError)),
		MergeWith4(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
			Delay[int](75*time.Millisecond)(Just(7, 8)),
		),
	)
	subscription := observable.Subscribe(PrintObserver[int]())
	time.Sleep(120 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 7
	// Next: 8
	// Error: assert.AnError general error for testing
}

func ExampleMergeWith5_ok() {
	observable := Pipe1(
		Delay[int](125*time.Millisecond)(Just(11, 12)),
		MergeWith5(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
			Delay[int](75*time.Millisecond)(Just(7, 8)),
			Delay[int](100*time.Millisecond)(Just(9, 10)),
		),
	)
	subscription := observable.Subscribe(PrintObserver[int]())
	time.Sleep(150 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 7
	// Next: 8
	// Next: 9
	// Next: 10
	// Next: 11
	// Next: 12
	// Completed
}

func ExampleMergeWith5_error() {
	observable := Pipe1(
		Delay[int](125*time.Millisecond)(Throw[int](assert.AnError)),
		MergeWith5(
			Just(1, 2),
			Delay[int](25*time.Millisecond)(Just(3, 4)),
			Delay[int](50*time.Millisecond)(Just(5, 6)),
			Delay[int](75*time.Millisecond)(Just(7, 8)),
			Delay[int](100*time.Millisecond)(Just(9, 10)),
		),
	)
	subscription := observable.Subscribe(PrintObserver[int]())
	time.Sleep(150 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Next: 7
	// Next: 8
	// Next: 9
	// Next: 10
	// Error: assert.AnError general error for testing
}

func ExampleMergeAll_ok() {
	observable := Pipe1(
		Just(
			Just(1),
			Delay[int](20*time.Millisecond)(Just(2)),
			Delay[int](40*time.Millisecond)(Just(3)),
		),
		MergeAll[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(60 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleMergeAll_error() {
	observable := Pipe1(
		Just(
			Delay[int](10*time.Millisecond)(Just(1)),
			Delay[int](50*time.Millisecond)(Throw[int](assert.AnError)),
			Delay[int](100*time.Millisecond)(Just(3)),
		),
		MergeAll[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(300 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Error: assert.AnError general error for testing
}

func ExampleMergeMap_ok() {
	observable := Pipe1(
		Just("a", "bb", "ccc"),
		MergeMap(func(item string) Observable[string] {
			return Delay[string](time.Duration(len(item)) * 50 * time.Millisecond)(Just(strings.ToUpper(item)))
		}),
	)
	subscription := observable.Subscribe(PrintObserver[string]())
	time.Sleep(200 * time.Millisecond)
	defer subscription.Unsubscribe()
	// Output:
	// Next: A
	// Next: BB
	// Next: CCC
	// Completed
}

func ExampleMergeMap_error() {
	observable := Pipe1(
		Just("a", "bb", "ccc"),
		MergeMap(func(item string) Observable[string] {
			if item == "bb" {
				return Throw[string](assert.AnError)
			}
			return Delay[string](time.Duration(len(item)) * 50 * time.Millisecond)(Just(strings.ToUpper(item)))
		}),
	)
	subscription := observable.Subscribe(PrintObserver[string]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleCombineLatestWith_ok() {
	observable1 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable := Pipe2(
		observable1,
		CombineLatestWith[int64](observable2),
		Map(func(snapshot lo.Tuple2[int64, int64]) []int64 {
			return []int64{snapshot.A, snapshot.B}
		}),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	time.Sleep(200 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 3]
	// Next: [1 4]
	// Next: [2 4]
	// Completed
}

func ExampleCombineLatestWith1_ok() {
	observable1 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable := Pipe1(
		CombineLatestWith1[int64](observable2)(observable1),
		Map(func(snapshot lo.Tuple2[int64, int64]) []int64 {
			return []int64{snapshot.A, snapshot.B}
		}),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())

	time.Sleep(200 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 3]
	// Next: [1 4]
	// Next: [2 4]
	// Completed
}

func ExampleCombineLatestWith2_ok() {
	observable1 := Delay[int64](150 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable3 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(5, 7, 50*time.Millisecond))

	combined := CombineLatestWith2[int64](observable2, observable3)(observable1)
	observable := Map(func(snapshot lo.Tuple3[int64, int64, int64]) []int64 {
		return []int64{snapshot.A, snapshot.B, snapshot.C}
	})(combined)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 4 6]
	// Next: [2 4 6]
	// Completed
}

func ExampleCombineLatestWith3_ok() {
	observable1 := Delay[int64](175 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable3 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(5, 7, 50*time.Millisecond))
	observable4 := Delay[int64](50 * time.Millisecond)(RangeWithInterval(7, 9, 50*time.Millisecond))

	combined := CombineLatestWith3[int64](observable2, observable3, observable4)(observable1)
	observable := Map(func(snapshot lo.Tuple4[int64, int64, int64, int64]) []int64 {
		return []int64{snapshot.A, snapshot.B, snapshot.C, snapshot.D}
	})(combined)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 4 6 8]
	// Next: [2 4 6 8]
	// Completed
}

func ExampleCombineLatestWith4_ok() {
	observable1 := Delay[int64](200 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable3 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(5, 7, 50*time.Millisecond))
	observable4 := Delay[int64](50 * time.Millisecond)(RangeWithInterval(7, 9, 50*time.Millisecond))
	observable5 := Delay[int64](75 * time.Millisecond)(RangeWithInterval(9, 11, 50*time.Millisecond))

	combined := CombineLatestWith4[int64](observable2, observable3, observable4, observable5)(observable1)
	observable := Map(func(snapshot lo.Tuple5[int64, int64, int64, int64, int64]) []int64 {
		return []int64{snapshot.A, snapshot.B, snapshot.C, snapshot.D, snapshot.E}
	})(combined)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 4 6 8 10]
	// Next: [2 4 6 8 10]
	// Completed
}

func ExampleCombineLatestAll_ok() {
	observable := Pipe1(
		Just(
			RangeWithInterval(1, 3, 40*time.Millisecond),
			RangeWithInterval(3, 5, 60*time.Millisecond),
			Delay[int64](25*time.Millisecond)(RangeWithInterval(5, 7, 100*time.Millisecond)),
		),
		CombineLatestAll[int64](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [2 4 5]
	// Next: [2 4 6]
	// Completed
}

func ExampleCombineLatestAll_error() {
	observable := Pipe1(
		Just(
			RangeWithInterval(1, 3, 50*time.Millisecond),
			Delay[int64](75*time.Millisecond)(Throw[int64](assert.AnError)),
			RangeWithInterval(5, 7, 50*time.Millisecond),
		),
		CombineLatestAll[int64](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())

	time.Sleep(200 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleCombineLatestAllAny_ok() {
	observable1 := Map(func(x int64) any { return x })(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := Of[any]("a", "b")
	observable3 := Delay[any](25 * time.Millisecond)(Of[any]("c", "d"))

	combined := Just(observable1, observable2, observable3)
	observable := CombineLatestAllAny()(combined)

	subscription := observable.Subscribe(PrintObserver[[]any]())

	time.Sleep(200 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 b d]
	// Next: [2 b d]
	// Completed
}

func ExampleCombineLatestAllAny_error() {
	observable1 := Map(func(x int64) any { return x })(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := Delay[any](75 * time.Millisecond)(Throw[any](assert.AnError))
	observable3 := Of[any]("a", "b")

	combined := Just(observable1, observable2, observable3)
	observable := CombineLatestAllAny()(combined)

	subscription := observable.Subscribe(PrintObserver[[]any]())

	time.Sleep(200 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleConcatWith_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		ConcatWith(Just(4, 5, 6)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleConcatWith_error() {
	observable := Pipe1(
		Just(1, 2, 3),
		ConcatWith(Throw[int](assert.AnError)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleConcatAll_ok() {
	observable := Pipe1(
		Just(
			Just(1, 2, 3),
			Just(4, 5, 6),
		),
		ConcatAll[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(30 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleConcatAll_error() {
	observable := Pipe1(
		Just(
			Just(1, 2, 3),
			Throw[int](assert.AnError),
			Just(4, 5, 6),
		),
		ConcatAll[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(30 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleStartWith_ok() {
	observable := Pipe1(
		Just(4, 5, 6),
		StartWith(1, 2, 3),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleStartWith_error() {
	observable := Pipe1(
		Throw[int](assert.AnError),
		StartWith(1, 2, 3),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleEndWith_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		EndWith(4, 5, 6),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleEndWith_error() {
	observable := Pipe1(
		Throw[int](assert.AnError),
		EndWith(1, 2, 3),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExamplePairwise_ok() {
	obsercable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Pairwise[int](),
	)

	subscription := obsercable.Subscribe(PrintObserver[[]int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Next: [2 3]
	// Next: [3 4]
	// Next: [4 5]
	// Completed
}

func ExamplePairwise_error() {
	observable := Pipe1(
		Throw[int](assert.AnError),
		Pairwise[int](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleRaceWith_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		RaceWith(
			Delay[int](50*time.Millisecond)(Just(4, 5, 6)),
			Delay[int](100*time.Millisecond)(Just(7, 8, 9)),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	time.Sleep(150 * time.Millisecond)

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleRaceWith_error() {
	observable := Race(
		Delay[int](50*time.Millisecond)(Throw[int](assert.AnError)),
		Delay[int](20*time.Millisecond)(Just(4, 5, 6)),
		Delay[int](100*time.Millisecond)(Just(7, 8, 9)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleZipWith_ok() {
	observable := ZipWith2[int](
		Range(10, 13),
		Range(20, 23),
	)(Just(1, 2, 3))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 20}
	// Next: {2 11 21}
	// Next: {3 12 22}
	// Completed
}

func ExampleZipWith_error() {
	observable := ZipWith2[int](
		Throw[int64](assert.AnError),
		Range(20, 23),
	)(Just(1, 2, 3))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipWith1_ok() {
	observable := ZipWith1[int](Range(10, 13))(Just(1, 2, 3))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple2[int, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10}
	// Next: {2 11}
	// Next: {3 12}
	// Completed
}

func ExampleZipWith1_error() {
	observable := ZipWith1[int](Throw[int64](assert.AnError))(Just(1, 2, 3))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple2[int, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipWith2_ok() {
	observable := ZipWith2[int](Range(10, 13), Range(20, 23))(Just(1, 2))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 20}
	// Next: {2 11 21}
	// Completed
}

func ExampleZipWith2_error() {
	observable := ZipWith2[int](Throw[int64](assert.AnError), Range(20, 23))(Just(1, 2))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipWith3_ok() {
	observable := ZipWith3[int](Range(10, 13), Range(20, 23), Range(30, 33))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple4[int, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 20 30}
	// Completed
}

func ExampleZipWith3_error() {
	observable := ZipWith3[int](Throw[int64](assert.AnError), Range(20, 23), Range(30, 33))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple4[int, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipWith4_ok() {
	observable := ZipWith4[int](Range(10, 13), Range(20, 23), Range(30, 33), Range(40, 43))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple5[int, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 20 30 40}
	// Completed
}

func ExampleZipWith4_error() {
	observable := ZipWith4[int](Throw[int64](assert.AnError), Range(20, 23), Range(30, 33), Range(40, 43))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple5[int, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipWith5_ok() {
	observable := ZipWith5[int](Range(10, 13), Range(20, 23), Range(30, 33), Range(40, 43), Range(50, 53))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple6[int, int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 20 30 40 50}
	// Completed
}

func ExampleZipWith5_error() {
	observable := ZipWith5[int](Throw[int64](assert.AnError), Range(20, 23), Range(30, 33), Range(40, 43), Range(50, 53))(Just(1))

	subscription := observable.Subscribe(PrintObserver[lo.Tuple6[int, int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZipAll_ok() {
	observable := Pipe1(
		Just(
			Range(1, 3),
			Range(10, 13),
			Range(100, 103),
		),
		ZipAll[int64](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 10 100]
	// Next: [2 11 101]
	// Completed
}

func ExampleZipAll_error() {
	observable := Pipe1(
		Just(
			Range(1, 3),
			Throw[int64](assert.AnError),
			Range(100, 3),
		),
		ZipAll[int64](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleAll_ok() {
	observable1 := Pipe1(
		Just(1, 2, 3, 4, 5),
		All(func(i int) bool { return i > 0 }),
	)

	subscription1 := observable1.Subscribe(PrintObserver[bool]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Just(1, 2, 3, 4, 5),
		All(func(i int) bool { return i%2 == 0 }),
	)

	subscription2 := observable2.Subscribe(PrintObserver[bool]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: true
	// Completed
	// Next: false
	// Completed
}

func ExampleAll_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		All(func(i int) bool { return i > 0 }),
	)

	subscription := observable.Subscribe(PrintObserver[bool]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleContains_ok() {
	observable1 := Pipe1(
		Just(1, 2, 3, 4, 5),
		Contains(func(i int) bool { return i < 0 }),
	)

	subscription1 := observable1.Subscribe(PrintObserver[bool]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Just(1, 2, 3, 4, 5),
		Contains(func(i int) bool { return i%2 == 0 }),
	)

	subscription2 := observable2.Subscribe(PrintObserver[bool]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: false
	// Completed
	// Next: true
	// Completed
}

func ExampleContains_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Contains(func(i int) bool { return i == 4 }),
	)

	subscription := observable.Subscribe(PrintObserver[bool]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleFind_ok() {
	observable1 := Pipe1(
		Just(1, 2, 3, 4, 5),
		Find(func(i int) bool { return i < 0 }),
	)

	subscription1 := observable1.Subscribe(PrintObserver[int]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Just(1, 2, 3, 4, 5),
		Find(func(i int) bool { return i%2 == 0 }),
	)

	subscription2 := observable2.Subscribe(PrintObserver[int]())
	defer subscription2.Unsubscribe()

	// Output:
	// Completed
	// Next: 2
	// Completed
}

func ExampleFind_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Find(func(i int) bool { return i == 4 }),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleIif_ok() {
	observable := Iif(
		func() bool {
			return true
		},
		Just(1, 2, 3),
		Just(4, 5, 6),
	)()

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleIif_error() {
	observable := Iif(
		func() bool {
			return false
		},
		Just(1, 2, 3),
		Throw[int](assert.AnError),
	)()

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleDefaultIfEmpty_ok() {
	observable1 := Pipe1(
		Just(1, 2, 3),
		DefaultIfEmpty(42),
	)

	subscription1 := observable1.Subscribe(PrintObserver[int]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Empty[int](),
		DefaultIfEmpty(42),
	)

	subscription2 := observable2.Subscribe(PrintObserver[int]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
	// Next: 42
	// Completed
}

func ExampleDefaultIfEmpty_error() {
	observable := Pipe1(
		Throw[int](assert.AnError),
		DefaultIfEmpty(42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleContextWithValue() {
	type contextValue struct{}

	observable := Pipe2(
		Just(1, 2, 3, 4, 5),
		ContextWithValue[int](contextValue{}, 42),
		Filter(func(i int) bool {
			return i%2 == 0
		}),
	)

	subscription := observable.Subscribe(
		OnNextWithContext(func(ctx context.Context, value int) {
			fmt.Printf("Next: %v\n", value)
			fmt.Printf("Next context value: %v\n", ctx.Value(contextValue{}))
		}),
	)
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next context value: 42
	// Next: 4
	// Next context value: 42
}

func ExampleNewObservable_ok() {
	observable := NewObservable(func(observer Observer[int]) Teardown {
		observer.Next(1)
		observer.Next(2)
		observer.Next(3)
		observer.Next(4)
		observer.Complete()

		return nil
	})

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleNewObservable_error() {
	observable := NewObservable(func(observer Observer[int]) Teardown {
		observer.Next(1)
		observer.Next(2)
		observer.Next(3)
		observer.Error(assert.AnError)
		observer.Next(4)

		return nil
	})

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleOf() {
	observable := Of(1, 2, 3)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleStart() {
	observable := Start(func() int {
		fmt.Println("Start!")
		return 42
	})

	subscription1 := observable.Subscribe(PrintObserver[int]())
	subscription2 := observable.Subscribe(PrintObserver[int]())

	subscription1.Wait() // Note: using .Wait() is not recommended.
	subscription2.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Start!
	// Next: 42
	// Completed
	// Start!
	// Next: 42
	// Completed
}

func ExampleJust() {
	observable := Just(1, 2, 3)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleTimer() {
	observable := Timer(10 * time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[time.Duration]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 10ms
	// Completed
}

func ExampleInterval() {
	observable := Interval(100 * time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[int64]())

	time.Sleep(250 * time.Millisecond)
	subscription.Unsubscribe() // "Completed" event is not transmitted

	// Output:
	// Next: 0
	// Next: 1
}

func ExampleIntervalWithInitial() {
	observable := IntervalWithInitial(50*time.Millisecond, 100*time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[int64]())

	time.Sleep(300 * time.Millisecond)
	subscription.Unsubscribe() // "Completed" event is not transmitted

	// Output:
	// Next: 0
	// Next: 1
	// Next: 2
}

func ExampleRange() {
	observable := Range(0, 5)

	subscription := observable.Subscribe(PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 0
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleRangeWithStep() {
	observable := RangeWithStep(0, 5, 0.5)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 0
	// Next: 0.5
	// Next: 1
	// Next: 1.5
	// Next: 2
	// Next: 2.5
	// Next: 3
	// Next: 3.5
	// Next: 4
	// Next: 4.5
	// Completed
}

func ExampleRangeWithInterval() {
	observable := RangeWithInterval(0, 5, 10*time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 0
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleRangeWithStepAndInterval() {
	observable := RangeWithStepAndInterval(0, 5, 0.5, 10*time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[float64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 0
	// Next: 0.5
	// Next: 1
	// Next: 1.5
	// Next: 2
	// Next: 2.5
	// Next: 3
	// Next: 3.5
	// Next: 4
	// Next: 4.5
	// Completed
}

func ExampleRepeat() {
	observable := Repeat(42, 3)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 42
	// Next: 42
	// Next: 42
	// Completed
}

func ExampleRepeatWithInterval() {
	observable := RepeatWithInterval(42, 3, 50*time.Millisecond)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	time.Sleep(200 * time.Millisecond)

	// Output:
	// Next: 42
	// Next: 42
	// Next: 42
	// Completed
}

func ExampleFromChannel() {
	ch := make(chan int, 10)
	observable := FromChannel(ch)

	subscription := observable.Subscribe(PrintObserver[int]())

	ch <- 1

	ch <- 2

	ch <- 3

	close(ch)

	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleFromSlice() {
	observable := FromSlice([]int{1, 2, 3})

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleEmpty() {
	observable := Empty[int]()

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Completed
}

func ExampleNever() {
	observable := Never()

	subscription := observable.Subscribe(PrintObserver[struct{}]())

	time.Sleep(10 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
}

func ExampleThrow() {
	observable := Throw[int](assert.AnError)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleDefer() {
	// will capture current date time
	observable1 := Of(time.Now())

	// will capture date time at the moment of subscription
	observable2 := Defer(func() Observable[time.Time] {
		return Of(time.Now())
	})

	subscription := Concat(observable1, observable2).Subscribe(NoopObserver[time.Time]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
}

func ExampleFuture_ok() {
	observable := Future(func() (int, error) {
		req, err := http.NewRequest("GET", "https://postman-echo.com/get", nil)
		if err != nil {
			return 0, err
		}

		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, err
		}

		defer res.Body.Close()

		// For some reason, removing the 2 following lines causes
		// the example to fail (see goleak).
		// See https://github.com/uber-go/goleak/issues/102
		_, _ = io.ReadAll(res.Body)

		defer http.DefaultClient.CloseIdleConnections()

		return 42, nil
	})

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 42
	// Completed
}

func ExampleFuture_error() {
	observable := Future(func() (int, error) {
		req, err := http.NewRequest("", "", nil)
		if err != nil {
			return 0, err
		}

		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return 0, err
		}

		defer res.Body.Close()

		// For some reason, removing the 2 following lines causes
		// the example to fail (see goleak).
		// See https://github.com/uber-go/goleak/issues/102
		_, _ = io.ReadAll(res.Body)

		defer http.DefaultClient.CloseIdleConnections()

		return 42, nil
	})

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Error: Get "": unsupported protocol scheme ""
}

func ExampleMerge_ok() {
	observable := Merge(
		RangeWithInterval(0, 2, 50*time.Millisecond),
		Pipe1(
			RangeWithInterval(10, 12, 50*time.Millisecond),
			Delay[int64](25*time.Millisecond),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 0
	// Next: 10
	// Next: 1
	// Next: 11
	// Completed
}

func ExampleMerge_error() {
	observable := Merge(
		RangeWithInterval(0, 2, 50*time.Millisecond),
		Pipe1(
			Throw[int64](assert.AnError),
			Delay[int64](75*time.Millisecond),
		),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 0
	// Error: assert.AnError general error for testing
}

func ExampleCombineLatest2_ok() {
	observable1 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)

	observable := Pipe1(
		CombineLatest2(
			observable1,
			observable2,
		),
		Map(func(snapshot lo.Tuple2[int64, int64]) []int64 {
			return []int64{snapshot.A, snapshot.B}
		}),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 3]
	// Next: [1 4]
	// Next: [2 4]
	// Completed
}

func ExampleCombineLatest2_error() {
	observable1 := NewObservable(func(observer Observer[int]) Teardown {
		go func() {
			time.Sleep(10 * time.Millisecond)
			observer.Next(1)
			observer.Error(assert.AnError)
		}()

		return nil
	})

	observable2 := NewObservable(func(observer Observer[int]) Teardown {
		go func() {
			observer.Next(2)
			observer.Complete()
		}()

		return nil
	})

	observable := Pipe1(
		CombineLatest2(
			observable1,
			observable2,
		),
		Map(func(snapshot lo.Tuple2[int, int]) []int {
			return []int{snapshot.A, snapshot.B}
		}),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(50 * time.Millisecond)

	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Error: assert.AnError general error for testing
}

func ExampleCombineLatest3_ok() {
	observable1 := Delay[int64](100 * time.Millisecond)(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := RangeWithInterval(3, 5, 50*time.Millisecond)
	observable3 := Delay[int64](25 * time.Millisecond)(RangeWithInterval(5, 7, 50*time.Millisecond))

	combined := CombineLatest3(observable1, observable2, observable3)
	observable := Map(func(snapshot lo.Tuple3[int64, int64, int64]) []int64 {
		return []int64{snapshot.A, snapshot.B, snapshot.C}
	})(combined)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 4 6]
	// Next: [2 4 6]
	// Completed
}

func ExampleCombineLatestAny_ok() {
	observable1 := Cast[int64, any]()(RangeWithInterval(1, 3, 40*time.Millisecond))
	observable2 := Cast[string, any]()(Just("a", "b"))
	observable3 := Delay[any](25 * time.Millisecond)(Just[any]("c", "d"))
	observable4 := Delay[any](60 * time.Millisecond)(Cast[int64, any]()(Range(100, 102)))

	combined := Just(observable1, observable2, observable3, observable4)
	observable := CombineLatestAllAny()(combined)

	subscription := observable.Subscribe(PrintObserver[[]any]())

	time.Sleep(220 * time.Millisecond)
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: [1 b d 100]
	// Next: [1 b d 101]
	// Next: [2 b d 101]
	// Completed
}

func ExampleCombineLatestAny_error() {
	observable1 := Cast[int64, any]()(RangeWithInterval(1, 3, 50*time.Millisecond))
	observable2 := Delay[any](75 * time.Millisecond)(Throw[any](assert.AnError))
	observable3 := Just[any]("a", "b")

	combined := Just(observable1, observable2, observable3)
	observable := CombineLatestAllAny()(combined)

	subscription := observable.Subscribe(PrintObserver[[]any]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip_ok() {
	observable := Zip(
		Range(1, 3),
		Range(10, 13),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 10]
	// Next: [2 11]
	// Completed
}

func ExampleZip_error() {
	observable := Zip(
		Range(1, 3),
		Throw[int64](assert.AnError),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip2_ok() {
	observable := Zip2(
		Range(0, 10),
		Skip[int64](1)(Range(0, 4)),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple2[int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {0 1}
	// Next: {1 2}
	// Next: {2 3}
	// Completed
}

func ExampleZip2_error() {
	observable := Zip2(
		Range(0, 10),
		Throw[int64](assert.AnError),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple2[int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip3_ok() {
	observable := Zip3(
		Range(1, 3),
		Range(10, 13),
		Range(100, 103),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 100}
	// Next: {2 11 101}
	// Completed
}

func ExampleZip3_error() {
	observable := Zip3(
		Range(1, 3),
		Throw[int64](assert.AnError),
		Range(100, 103),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple3[int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip4_ok() {
	observable := Zip4(
		Range(1, 3),
		Range(10, 13),
		Range(100, 103),
		Range(1000, 1003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple4[int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 100 1000}
	// Next: {2 11 101 1001}
	// Completed
}

func ExampleZip4_error() {
	observable := Zip4(
		Range(1, 3),
		Throw[int64](assert.AnError),
		Range(100, 103),
		Range(1000, 1003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple4[int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip5_ok() {
	observable := Zip5(
		Range(1, 3),
		Range(10, 13),
		Range(100, 103),
		Range(1000, 1003),
		Range(10000, 10003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple5[int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 100 1000 10000}
	// Next: {2 11 101 1001 10001}
	// Completed
}

func ExampleZip5_error() {
	observable := Zip5(
		Range(1, 3),
		Throw[int64](assert.AnError),
		Range(100, 103),
		Range(1000, 1003),
		Range(10000, 10003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple5[int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleZip6_ok() {
	observable := Zip6(
		Range(1, 3),
		Range(10, 13),
		Range(100, 103),
		Range(1000, 1003),
		Range(10000, 10003),
		Range(100000, 100003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple6[int64, int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: {1 10 100 1000 10000 100000}
	// Next: {2 11 101 1001 10001 100001}
	// Completed
}

func ExampleZip6_error() {
	observable := Zip6(
		Range(1, 3),
		Throw[int64](assert.AnError),
		Range(100, 103),
		Range(1000, 1003),
		Range(10000, 10003),
		Range(100000, 100003),
	)

	subscription := observable.Subscribe(PrintObserver[lo.Tuple6[int64, int64, int64, int64, int64, int64]]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleConcat_ok() {
	observable := Concat(
		Just(1, 2, 3),
		Just(4, 5, 6),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleConcat_error() {
	observable := Concat(
		Just(1, 2, 3),
		Throw[int](assert.AnError),
		Just(4, 5, 6),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleRace_ok() {
	observable := Race(
		Delay[int](75*time.Millisecond)(Just(1, 2, 3)),
		Delay[int](25*time.Millisecond)(Just(4, 5, 6)),
		Delay[int](100*time.Millisecond)(Just(7, 8, 9)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(50 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleRace_error() {
	observable := Race(
		Delay[int](75*time.Millisecond)(Just(1, 2, 3)),
		Delay[int](25*time.Millisecond)(Throw[int](assert.AnError)),
		Delay[int](100*time.Millisecond)(Just(7, 8, 9)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(50 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleAmb_ok() {
	observable := Amb(
		Delay[int](100*time.Millisecond)(Just(1, 2, 3)),
		Delay[int](25*time.Millisecond)(Just(4, 5, 6)),
		Delay[int](50*time.Millisecond)(Just(7, 8, 9)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(150 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleAmb_error() {
	observable := Amb(
		Delay[int](25*time.Millisecond)(Throw[int](assert.AnError)),
		Delay[int](50*time.Millisecond)(Just(4, 5, 6)),
		Delay[int](100*time.Millisecond)(Just(7, 8, 9)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(75 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

// func ExampleRandIntN() {
// 	observable := RandIntN(10, 5)

// 	subscription := observable.Subscribe(PrintObserver[int]())
// 	defer subscription.Unsubscribe()

// 	// Output:
// 	// Next: 0
// 	// Next: 3
// 	// Next: 7
// 	// Next: 1
// 	// Next: 9
// 	// Completed
// }

// func ExampleRandFloat64() {
// 	observable := RandFloat64(3)

// 	subscription := observable.Subscribe(PrintObserver[float64]())
// 	defer subscription.Unsubscribe()

// 	// Output:
// 	// Next: 0.123456
// 	// Next: 0.789012
// 	// Next: 0.345678
// 	// Completed
// }

func ExampleCatch() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)
			observer.Complete()

			return nil
		}),
		Catch(func(err error) Observable[int] {
			return Of(4, 5, 6)
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleOnErrorResumeNextWith() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)
			observer.Complete()

			return nil
		}),
		OnErrorResumeNextWith(Of(4, 5, 6)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExampleOnErrorReturn() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)
			observer.Complete()

			return nil
		}),
		OnErrorReturn(42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 42
	// Completed
}

func ExampleRetryWithConfig() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)
			observer.Complete()

			return nil
		}),
		RetryWithConfig[int](RetryConfig{
			MaxRetries: 1,
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleThrowIfEmpty() {
	observable := Pipe1(
		Empty[int](),
		ThrowIfEmpty[int](func() error {
			return errors.New("empty")
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: empty
}

func ExampleDoWhile() {
	i := 0

	observable := Pipe1(
		Just(1, 2, 3),
		DoWhile[int](func() bool {
			i++
			return i < 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleWhile() {
	i := 0

	observable := Pipe1(
		Just(1, 2, 3),
		While[int](func() bool {
			i++
			return i < 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleFilter_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Filter(func(i int) bool {
			return i%2 == 0
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 4
	// Completed
}

func ExampleFilter_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Filter(func(i int) bool {
			return i%2 == 0
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Error: assert.AnError general error for testing
}

func ExampleDistinct_ok() {
	observable := Pipe1(
		Just(1, 1, 2, 2, 3, 3, 4, 4, 5, 5),
		Distinct[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleDistinct_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(1)
			observer.Next(2)
			observer.Next(2)
			observer.Next(3)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)
			observer.Next(4)

			return nil
		}),
		Distinct[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleIgnoreElements_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		IgnoreElements[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Completed
}

func ExampleIgnoreElements_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		IgnoreElements[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleSkip_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Skip[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleSkip_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Skip[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleSkipWhile_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		SkipWhile(func(v int) bool {
			return v > 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleSkipWhile_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		SkipWhile(func(v int) bool {
			return v > 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleSkipLast_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		SkipLast[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleSkipLast_empty() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		SkipLast[int](10),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Completed
}

func ExampleSkipLast_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		SkipLast[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Error: assert.AnError general error for testing
}

func ExampleSkipUntil_ok() {
	observable := Pipe1(
		RangeWithInterval(0, 5, 40*time.Millisecond),
		SkipUntil[int64](Interval(100*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleSkipUntil_empty() {
	observable := Pipe1(
		RangeWithInterval(0, 5, 10*time.Millisecond),
		SkipUntil[int64](Interval(100*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Completed
}

func ExampleSkipUntil_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				time.Sleep(30 * time.Millisecond)
				observer.Next(1)
				time.Sleep(30 * time.Millisecond)
				observer.Next(2)
				time.Sleep(30 * time.Millisecond)
				observer.Next(3)
				observer.Error(assert.AnError)
				observer.Next(4)
			}()

			return nil
		}),
		SkipUntil[int](Interval(45*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleTake_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Take[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleTake_error1() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Take[int](5),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleTake_error2() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Take[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleTakeLast_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		TakeLast[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleTakeLast_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		TakeLast[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleTakeWhile_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		TakeWhile(func(n int) bool {
			return n < 3
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleTakeWhile_error1() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		TakeWhile(func(n int) bool {
			return n < 5
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleTakeWhile_error2() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		TakeWhile(func(n int) bool {
			return n < 3
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleTakeUntil_ok() {
	observable := Pipe1(
		RangeWithInterval(0, 5, 40*time.Millisecond),
		TakeUntil[int64](Interval(100*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 0
	// Next: 1
	// Completed
}

func ExampleTakeUntil_empty() {
	observable := Pipe1(
		RangeWithInterval(0, 5, 50*time.Millisecond),
		TakeUntil[int64](Interval(10*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Completed
}

func ExampleTakeUntil_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				time.Sleep(20 * time.Millisecond)
				observer.Next(1)
				time.Sleep(20 * time.Millisecond)
				observer.Next(2)
				time.Sleep(20 * time.Millisecond)
				observer.Next(3)
				observer.Error(assert.AnError)
				observer.Next(4)
			}()

			return nil
		}),
		TakeUntil[int](Interval(50*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Completed
}

func ExampleHead_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Head[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Completed
}

func ExampleHead_error() {
	observable1 := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Head[int](),
	)

	subscription1 := observable1.Subscribe(PrintObserver[int]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Throw[int](assert.AnError), // no item transmitted
		Head[int](),
	)

	subscription2 := observable2.Subscribe(PrintObserver[int]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: 1
	// Completed
	// Error: assert.AnError general error for testing
}

func ExampleTail_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Tail[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Completed
}

func ExampleTail_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Tail[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleFirst_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		First(func(n int) bool {
			return n > 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Completed
}

func ExampleFirst_error() {
	observable1 := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		First(func(n int) bool {
			return n > 2
		}),
	)

	subscription1 := observable1.Subscribe(PrintObserver[int]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Throw[int](assert.AnError), // no item transmitted
		First(func(n int) bool {
			return n > 2
		}),
	)

	subscription2 := observable2.Subscribe(PrintObserver[int]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: 3
	// Completed
	// Error: assert.AnError general error for testing
}

func ExampleLast_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Last(func(n int) bool {
			return n > 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Completed
}

func ExampleLast_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Last(func(n int) bool {
			return n > 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleElementAt_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ElementAt[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Completed
}

func ExampleElementAt_notFound() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ElementAt[int](10),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: ro.ElementAt: nth element not found
}

func ExampleElementAt_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		ElementAt[int](10),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleElementAtOrDefault_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ElementAtOrDefault(2, 100),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Completed
}

func ExampleElementAtOrDefault_notFound() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ElementAtOrDefault(10, 100),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 100
	// Completed
}

func ExampleElementAtOrDefault_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		ElementAtOrDefault(10, 100),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleAverage_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Average[int](),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Completed
}

func ExampleAverage_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Average[int](),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleCount_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Count[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Completed
}

func ExampleCount_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Count[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleSum_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 15
	// Completed
}

func ExampleSum_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleRound_ok() {
	observable := Pipe1(
		Just[float64](1, 2, 3, 4, 5),
		Round(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleRound_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[float64]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Round(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleMin_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Min[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Completed
}

func ExampleMin_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Min[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleMax_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Max[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Completed
}

func ExampleMax_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Max[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleClamp_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Clamp(2, 4),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 4
	// Completed
}

func ExampleClamp_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Clamp(2, 4),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleAbs_ok() {
	observable := Pipe1(
		Just[float64](-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5),
		Abs(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 5
	// Next: 4
	// Next: 3
	// Next: 2
	// Next: 1
	// Next: 0
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleAbs_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[float64]) Teardown {
			observer.Next(-3)
			observer.Next(-2)
			observer.Next(-1)
			observer.Next(0)
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Abs(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 3
	// Next: 2
	// Next: 1
	// Next: 0
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleFloor_ok() {
	observable := Pipe1(
		Just(1.1, 2.4, 3.5, 4.9, 5.0),
		Floor(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleFloor_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[float64]) Teardown {
			observer.Next(1.1)
			observer.Next(2.5)
			observer.Next(3.9)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Floor(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleCeil_ok() {
	observable := Pipe1(
		Just(1.1, 2.4, 3.5, 4.9, 5.0),
		Ceil(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 5
	// Completed
}

func ExampleCeil_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[float64]) Teardown {
			observer.Next(1.1)
			observer.Next(2.5)
			observer.Next(3.9)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Ceil(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 3
	// Next: 4
	// Error: assert.AnError general error for testing
}

func ExampleTrunc_ok() {
	observable := Pipe1(
		Just(1.1, 2.4, 3.5, 4.9, 5.0),
		Trunc(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Completed
}

func ExampleTrunc_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[float64]) Teardown {
			observer.Next(1.1)
			observer.Next(2.5)
			observer.Next(3.9)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Trunc(),
	)

	subscription := observable.Subscribe(PrintObserver[float64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleReduce_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Reduce(func(agg, current int) int {
			return agg + current
		}, 42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 57
	// Completed
}

func ExampleReduce_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Reduce(func(agg, current int) int {
			return agg + current
		}, 42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleToSlice_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ToSlice[int](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: [1 2 3 4 5]
	// Completed
}

func ExampleToSlice_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		ToSlice[int](),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleToMap_ok() {
	mapper := func(v int) (string, string) {
		return strconv.FormatInt(int64(v), 10), strconv.FormatInt(int64(v), 10)
	}

	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		ToMap(mapper),
	)

	subscription := observable.Subscribe(PrintObserver[map[string]string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: map[1:1 2:2 3:3 4:4 5:5]
	// Completed
}

func ExampleToMap_error() {
	mapper := func(v int) (string, string) {
		return strconv.FormatInt(int64(v), 10), strconv.FormatInt(int64(v), 10)
	}

	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		ToMap(mapper),
	)

	subscription := observable.Subscribe(PrintObserver[map[string]string]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleToChannel_ok() {
	observable := Pipe3(
		Just(1, 2, 3, 4, 5),
		ToChannel[int](42),
		Map(lo.ChannelToSlice[Notification[int]]),
		Flatten[Notification[int]](),
	)

	subscription := observable.Subscribe(PrintObserver[Notification[int]]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: Next(1)
	// Next: Next(2)
	// Next: Next(3)
	// Next: Next(4)
	// Next: Next(5)
	// Next: Complete()
	// Completed
}

func ExampleToChannel_error() {
	observable := Pipe3(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		ToChannel[int](42),
		Map(lo.ChannelToSlice[Notification[int]]),
		Flatten[Notification[int]](),
	)

	subscription := observable.Subscribe(PrintObserver[Notification[int]]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: Next(1)
	// Next: Next(2)
	// Next: Next(3)
	// Next: Error(assert.AnError general error for testing)
	// Completed
}

func ExampleMap_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 4
	// Next: 6
	// Next: 8
	// Next: 10
	// Completed
}

func ExampleMap_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Map(func(x int) int {
			return x * 2
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 4
	// Next: 6
	// Error: assert.AnError general error for testing
}

func ExampleMapTo_ok() {
	observable := Pipe2(
		Just(1, 2, 3, 4, 5),
		MapTo[int]("Hey!"),
		Take[string](3),
	)

	subscription := observable.Subscribe(PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hey!
	// Next: Hey!
	// Next: Hey!
	// Completed
}

func ExampleMapTo_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		MapTo[int]("Hey!"),
	)

	subscription := observable.Subscribe(PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hey!
	// Next: Hey!
	// Next: Hey!
	// Error: assert.AnError general error for testing
}

func ExampleMapErr_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		MapErr(func(item int) (string, error) {
			return "Hey!", nil
		}),
	)

	subscription := observable.Subscribe(PrintObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Hey!
	// Next: Hey!
	// Next: Hey!
	// Completed
}

func ExampleMapErr_error() {
	observable1 := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		MapErr(func(item int) (string, error) {
			return "Hey!", nil
		}),
	)

	subscription1 := observable1.Subscribe(PrintObserver[string]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Just(1, 2, 3, 4, 5),
		MapErr(func(item int) (string, error) {
			if item == 2 {
				return "Hey!", assert.AnError
			}

			return "Hey!", nil
		}),
	)

	subscription2 := observable2.Subscribe(PrintObserver[string]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: Hey!
	// Next: Hey!
	// Next: Hey!
	// Error: assert.AnError general error for testing
	// Next: Hey!
	// Error: assert.AnError general error for testing
}

func ExampleFlatMap_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		FlatMap(func(item int) Observable[int] {
			return Just(item, item)
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 1
	// Next: 2
	// Next: 2
	// Next: 3
	// Next: 3
	// Completed
}

func ExampleFlatMap_error() {
	observable1 := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		FlatMap(func(item int) Observable[int] {
			return Just(item, item)
		}),
	)

	subscription1 := observable1.Subscribe(PrintObserver[int]())
	defer subscription1.Unsubscribe()

	observable2 := Pipe1(
		Just(1, 2, 3),
		FlatMap(func(item int) Observable[int] {
			if item == 2 {
				return Throw[int](assert.AnError)
			}

			return Just(item, item)
		}),
	)

	subscription2 := observable2.Subscribe(PrintObserver[int]())
	defer subscription2.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 1
	// Next: 2
	// Next: 2
	// Next: 3
	// Next: 3
	// Error: assert.AnError general error for testing
	// Next: 1
	// Next: 1
	// Error: assert.AnError general error for testing
}

func ExampleScan_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Scan(func(agg, current int) int {
			return agg + current
		}, 42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 43
	// Next: 45
	// Next: 48
	// Next: 52
	// Next: 57
	// Completed
}

func ExampleScan_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Scan(func(agg, current int) int {
			return agg + current
		}, 42),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 43
	// Next: 45
	// Next: 48
	// Error: assert.AnError general error for testing
}

func ExampleGroupBy_ok() {
	odd := func(v int64) bool { return v%2 == 0 }

	observable := Pipe2(
		RangeWithInterval(1, 5, 10*time.Millisecond),
		GroupBy(odd),
		MergeAll[int64](),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 4
	// Completed
}

func ExampleGroupBy_error() {
	odd := func(v int) bool { return v%2 == 0 }

	observable := Pipe2(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			time.Sleep(5 * time.Millisecond)
			observer.Next(2)
			time.Sleep(5 * time.Millisecond)
			observer.Next(3)
			time.Sleep(5 * time.Millisecond)
			observer.Error(assert.AnError)
			time.Sleep(5 * time.Millisecond)
			observer.Next(4)

			return nil
		}),
		GroupBy(odd),
		MergeAll[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleBufferWhen_ok() {
	observable := Pipe1(
		Interval(30*time.Millisecond),
		BufferWhen[int64](Interval(100*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	time.Sleep(250 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [0 1 2]
	// Next: [3 4 5]
}

func ExampleBufferWhen_error() {
	observable := Pipe1(
		Throw[int64](assert.AnError),
		BufferWhen[int64](Interval(50*time.Millisecond)),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())
	defer subscription.Unsubscribe()

	time.Sleep(200 * time.Millisecond)

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleBufferWithTimeOrCount_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		BufferWithTimeOrCount[int](2, 100*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(10 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Next: [3 4]
	// Next: [5]
	// Completed
}

func ExampleBufferWithTimeOrCount_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				observer.Next(1)
				observer.Next(2)
				observer.Next(3)
				observer.Error(assert.AnError)
				observer.Next(4)
			}()

			return nil
		}),
		BufferWithTimeOrCount[int](2, 100*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(10 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Error: assert.AnError general error for testing
}

func ExampleBufferWithCount_ok() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		BufferWithCount[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(10 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Next: [3 4]
	// Next: [5]
	// Completed
}

func ExampleBufferWithCount_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				observer.Next(1)
				observer.Next(2)
				observer.Next(3)
				observer.Error(assert.AnError)
				observer.Next(4)
			}()

			return nil
		}),
		BufferWithCount[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(10 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2]
	// Error: assert.AnError general error for testing
}

// Commented because i get a weired conflict with other tests.
func ExampleBufferWithTime_ok() {
	observable := Pipe1(
		RangeWithInterval(1, 6, 20*time.Millisecond),
		BufferWithTime[int64](70*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[[]int64]())

	time.Sleep(200 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2 3]
	// Next: [4 5]
	// Completed
}

func ExampleBufferWithTime_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				observer.Next(1)
				time.Sleep(10 * time.Millisecond)
				observer.Next(2)
				time.Sleep(10 * time.Millisecond)
				observer.Next(3)

				time.Sleep(200 * time.Millisecond)
				// 1 empty buffer

				observer.Next(4)
				observer.Error(assert.AnError)
				observer.Next(5)
			}()

			return nil
		}),
		BufferWithTime[int](100*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[[]int]())

	time.Sleep(300 * time.Millisecond)
	subscription.Unsubscribe()

	// Output:
	// Next: [1 2 3]
	// Next: []
	// Error: assert.AnError general error for testing
}

func ExampleTap_ok() {
	observable := Pipe1(
		Range(1, 4),
		Tap(
			func(value int64) {
				fmt.Printf("Next: %v\n", value)
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)

	subscription := observable.Subscribe(NoopObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleTap_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Tap(
			func(value int) {
				fmt.Printf("Next: %v\n", value)
			},
			func(err error) {
				fmt.Printf("Error: %s\n", err.Error())
			},
			func() {
				fmt.Printf("Completed\n")
			},
		),
	)

	subscription := observable.Subscribe(NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleTapOnNext_ok() {
	observable := Pipe1(
		Range(1, 4),
		TapOnNext(func(v int64) { fmt.Println("Next:", v) }),
	)

	subscription := observable.Subscribe(NoopObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
}

func ExampleTapOnNext_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int64]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		TapOnNext(func(v int64) { fmt.Println("Next:", v) }),
	)

	subscription := observable.Subscribe(NoopObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
}

func ExampleTapOnError_ok() {
	observable := Pipe1(
		Range(1, 4),
		TapOnError[int64](func(err error) { fmt.Printf("Error: %s\n", err.Error()) }),
	)

	subscription := observable.Subscribe(NoopObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
}

func ExampleTapOnError_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		TapOnError[int](func(err error) { fmt.Printf("Error: %s\n", err.Error()) }),
	)

	subscription := observable.Subscribe(NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Error: assert.AnError general error for testing
}

func ExampleTapOnComplete_ok() {
	observable := Pipe1(
		Range(1, 4),
		TapOnComplete[int64](func() { fmt.Printf("Completed") }),
	)

	subscription := observable.Subscribe(NoopObserver[int64]())
	defer subscription.Unsubscribe()

	// Output:
	// Completed
}

func ExampleTapOnComplete_error() {
	observable := Pipe2(
		Throw[int](assert.AnError),
		Delay[int](10*time.Millisecond),
		TapOnComplete[int](func() { fmt.Printf("Completed") }),
	)

	subscription := observable.Subscribe(NoopObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
}

func ExampleTimeInterval() {
	observable := Pipe1(
		RangeWithInterval(0, 3, 10*time.Millisecond),
		TimeInterval[int64](),
	)

	subscription := observable.Subscribe(NoopObserver[IntervalValue[int64]]())
	defer subscription.Unsubscribe()
}

func ExampleTimestamp() {
	observable := Pipe1(
		RangeWithInterval(0, 3, 10*time.Millisecond),
		Timestamp[int64](),
	)

	subscription := observable.Subscribe(NoopObserver[TimestampValue[int64]]())
	defer subscription.Unsubscribe()
}

func ExampleDelay_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		Delay[int](10*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleDelay_cancel() {
	observable := Pipe1(
		Of(1),
		Delay[int](100*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[int]())

	time.Sleep(50 * time.Millisecond)
	subscription.Unsubscribe() // canceled before first message

	// Output:
}

func ExampleDelay_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Delay[int](10*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleRepeatWith_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		RepeatWith[int](3),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 1
	// Next: 2
	// Next: 3
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleRepeatWith_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		RepeatWith[int](3),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExampleTimeout_ok() {
	observable := Pipe1(
		Range(1, 4),
		Timeout[int64](20*time.Millisecond),
	)

	subscription := observable.Subscribe(PrintObserver[int64]())
	subscription.Wait() // Note: using .Wait() is not recommended.
	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleTimeout_error() {
	subscription := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			go func() {
				observer.Next(1)
				time.Sleep(100 * time.Millisecond)
				observer.Next(2)
				time.Sleep(100 * time.Millisecond)
				observer.Next(3)
				observer.Error(assert.AnError)
				observer.Next(4)
			}()
			return nil
		}),
		Timeout[int](50*time.Millisecond),
	).Subscribe(PrintObserver[int]())

	subscription.Wait() // Note: using .Wait() is not recommended.

	// Output:
	// Next: 1
	// Error: ro.Timeout: timeout after 50ms
}

func ExampleMaterialize_ok() {
	observable := Pipe1(
		Just(1, 2, 3),
		Materialize[int](),
	)

	subscription := observable.Subscribe(PrintObserver[Notification[int]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Next(1)
	// Next: Next(2)
	// Next: Next(3)
	// Next: Complete()
	// Completed
}

func ExampleMaterialize_error() {
	observable := Pipe1(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Next(3)
			observer.Error(assert.AnError)
			observer.Next(4)

			return nil
		}),
		Materialize[int](),
	)

	subscription := observable.Subscribe(PrintObserver[Notification[int]]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: Next(1)
	// Next: Next(2)
	// Next: Next(3)
	// Next: Error(assert.AnError general error for testing)
	// Completed
}

func ExampleDematerialize_ok() {
	observable := Pipe1(
		Just(
			Notification[int]{Kind: KindNext, Value: 1, Err: nil},
			Notification[int]{Kind: KindNext, Value: 2, Err: nil},
			Notification[int]{Kind: KindNext, Value: 3, Err: nil},
			Notification[int]{Kind: KindComplete, Value: 0, Err: nil},
		),
		Dematerialize[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Completed
}

func ExampleDematerialize_error() {
	observable := Pipe1(
		Just(
			Notification[int]{Kind: KindNext, Value: 1, Err: nil},
			Notification[int]{Kind: KindNext, Value: 2, Err: nil},
			Notification[int]{Kind: KindNext, Value: 3, Err: nil},
			Notification[int]{Kind: KindError, Value: 0, Err: assert.AnError},
		),
		Dematerialize[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 1
	// Next: 2
	// Next: 3
	// Error: assert.AnError general error for testing
}

func ExamplePipe() {
	observable := Pipe[int, int](
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Skip[int](2),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 24
	// Completed
}

func ExamplePipe1() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 15
	// Completed
}

func ExamplePipe2() {
	observable := Pipe2(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 30
	// Completed
}

func ExamplePipe3() {
	observable := Pipe3(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Skip[int](2),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 24
	// Completed
}

func ExamplePipe4() {
	observable := Pipe4(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Skip[int](2),
		Take[int](2),
		Sum[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 14
	// Completed
}

func ExamplePipe5() {
	observable := Pipe5(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Skip[int](2),
		Take[int](2),
		Sum[int](),
		Max[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 14
	// Completed
}

func ExamplePipe6() {
	observable := Pipe6(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Skip[int](2),
		Take[int](2),
		Sum[int](),
		Map(func(x int) int {
			return x / 2
		}),
		Max[int](),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 7
	// Completed
}

func ExamplePipeOp() {
	observable := Pipe1(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x + 1
		}),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 3
	// Next: 4
	// Next: 5
	// Next: 6
	// Completed
}

func ExamplePipeOp4() {
	observable := Pipe3(
		Just(1, 2, 3, 4, 5),
		Map(func(x int) int {
			return x * 2
		}),
		Filter(func(x int) bool {
			return x%2 == 0
		}),
		Take[int](2),
	)

	subscription := observable.Subscribe(PrintObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// Next: 2
	// Next: 4
	// Completed
}

func ExampleNewAsyncSubject() {
	subject := NewAsyncSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	sub := Pipe1(
		subject.AsObservable(),
		Delay[int](25*time.Millisecond),
	).Subscribe(PrintObserver[int]())
	defer sub.Unsubscribe()

	subject.Next(456) // nothing logged

	subject.Complete() // 456 logged by both subscribers

	time.Sleep(50 * time.Millisecond)

	subject.Next(789)                       // nothing logged
	subject.Subscribe(PrintObserver[int]()) // 456 logged by both subscribers

	// Output:
	// Next: 456
	// Completed
	// Next: 456
	// Completed
	// Next: 456
	// Completed
}

func ExampleNewAsyncSubject_error() {
	subject := NewAsyncSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	subject.Subscribe(PrintObserver[int]())

	subject.Next(456) // nothing logged

	subject.Error(assert.AnError) // error logged by both subscribers

	subject.Subscribe(PrintObserver[int]()) // error logged by last subscriber

	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	// Output:
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
}

func ExampleNewAsyncSubject_empty() {
	subject := NewAsyncSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	// Output:
	// Completed
	// Completed
}

func ExampleNewBehaviorSubject() {
	subject := NewBehaviorSubject(42)

	subject.Subscribe(PrintObserver[int]()) // 42 logged by first subscriber

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // 123 logged by second subscriber

	subject.Next(456) // 123 logged by second subscriber

	subject.Complete() // 456 logged by both subscribers

	subject.Next(789)                       // nothing logged
	subject.Subscribe(PrintObserver[int]()) // nothing logged

	// Output:
	// Next: 42
	// Next: 123
	// Next: 123
	// Next: 456
	// Next: 456
	// Completed
	// Completed
	// Completed
}

func ExampleNewBehaviorSubject_error() {
	subject := NewBehaviorSubject(42)

	subject.Subscribe(PrintObserver[int]()) // 42 logged by first subscriber

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // 123 logged by second subscriber

	subject.Next(456) // nothing logged

	subject.Error(assert.AnError) // error logged by both subscribers

	subject.Subscribe(PrintObserver[int]()) // error logged by last subscriber

	subject.Next(789) // nothing logged

	// Output:
	// Next: 42
	// Next: 123
	// Next: 123
	// Next: 456
	// Next: 456
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
}

func ExampleNewBehaviorSubject_empty() {
	subject := NewBehaviorSubject(42)

	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]()) // nothing logged
	subject.Subscribe(PrintObserver[int]()) // nothing logged

	subject.Next(123) // nothing logged

	// Output:
	// Completed
	// Completed
}

func ExampleNewPublishSubject() {
	subject := NewPublishSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]())

	subject.Next(456) // 456 logged by both subscribers

	subject.Complete()

	subject.Next(789) // nothing logged

	// Output:
	// Next: 123
	// Next: 456
	// Next: 456
	// Completed
	// Completed
}

func ExampleNewPublishSubject_error() {
	subject := NewPublishSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]())

	subject.Next(456) // 456 logged by both subscribers

	subject.Error(assert.AnError) // error logged by both subscribers

	subject.Subscribe(PrintObserver[int]()) // error logged by last subscriber

	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	// Output:
	// Next: 123
	// Next: 456
	// Next: 456
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
}

func ExampleNewPublishSubject_empty() {
	subject := NewPublishSubject[int]()

	subject.Subscribe(PrintObserver[int]())

	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	// Output:
	// Completed
	// Completed
}

func ExampleNewReplaySubject() {
	subject := NewReplaySubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // 123 logged by second subscriber

	subject.Next(456) // 456 logged by both subscriber

	subject.Complete()

	subject.Subscribe(PrintObserver[int]()) // 123 and 456 logged by third subscriber

	subject.Next(789) // nothing logged

	// Output:
	// Next: 123
	// Next: 123
	// Next: 456
	// Next: 456
	// Completed
	// Completed
	// Next: 123
	// Next: 456
	// Completed
}

func ExampleNewReplaySubject_error() {
	subject := NewReplaySubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // 123 logged by second subscriber

	subject.Next(456) // 456 logged by both subscriber

	subject.Error(assert.AnError) // error logged by both subscribers

	subject.Subscribe(PrintObserver[int]()) // error logged by last subscriber

	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	// Output:
	// Next: 123
	// Next: 123
	// Next: 456
	// Next: 456
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
	// Next: 123
	// Next: 456
	// Error: assert.AnError general error for testing
}

func ExampleNewReplaySubject_empty() {
	subject := NewReplaySubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	// Output:
	// Completed
	// Completed
}

func ExampleNewReplaySubject_overflow() {
	subject := NewReplaySubject[int](2)

	subject.Next(123)  // nothing logged
	subject.Next(456)  // nothing logged
	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]()) // 456 and 789 logged

	// Output:
	// Next: 456
	// Next: 789
	// Completed
}

func ExampleNewUnicastSubject() {
	subject := NewUnicastSubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // error

	subject.Next(456) // 456 logged by both subscriber

	subject.Complete()

	subject.Subscribe(PrintObserver[int]()) // 123 and 456 logged by third subscriber

	subject.Next(789) // 789 logged by third subscriber

	// Output:
	// Next: 123
	// Error: ro.UnicastSubject: a single subscriber accepted
	// Next: 456
	// Completed
	// Completed
}

func ExampleNewUnicastSubject_error() {
	subject := NewUnicastSubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // 123 logged by first subscriber

	subject.Subscribe(PrintObserver[int]()) // 123 logged by second subscriber

	subject.Next(456) // 456 logged by both subscriber

	subject.Error(assert.AnError) // error logged by both subscribers

	subject.Subscribe(PrintObserver[int]()) // error logged by last subscriber

	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	// Output:
	// Next: 123
	// Error: ro.UnicastSubject: a single subscriber accepted
	// Next: 456
	// Error: assert.AnError general error for testing
	// Error: assert.AnError general error for testing
}

func ExampleNewUnicastSubject_empty() {
	subject := NewUnicastSubject[int](42)

	subject.Subscribe(PrintObserver[int]())

	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]())

	subject.Next(123) // nothing logged

	// Output:
	// Completed
	// Completed
}

func ExampleNewUnicastSubject_overflow() {
	subject := NewUnicastSubject[int](2)

	subject.Next(123)  // nothing logged
	subject.Next(456)  // nothing logged
	subject.Next(789)  // nothing logged
	subject.Complete() // nothing logged

	subject.Subscribe(PrintObserver[int]()) // 456 and 789 logged

	// Output:
	// Completed
}
