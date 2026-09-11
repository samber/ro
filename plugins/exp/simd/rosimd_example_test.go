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

package rosimd_test

import (
	"fmt"
	"simd"

	"github.com/samber/ro"
	rosimd "github.com/samber/ro/plugins/exp/simd"
)

func ExampleVectorizeInt8() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, int8](
		ro.Just[int8](1, 2, 3, 4, 5),
		rosimd.VectorizeInt8,
		rosimd.AddInt8(rosimd.BroadcastInt8(10)),
		rosimd.ReduceSumInt8,
	)

	sub := obs.Subscribe(ro.OnNext(func(total int8) {
		fmt.Println(total)
	}))
	defer sub.Unsubscribe()

	// Output: 65
}

// Leaving vector space is done with core operators: there is no devectorize.
func ExampleAddInt8() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1
	// 2
	// 3
}

// Methods chain inside ro.Map and need no type arguments, unlike the operators.
func ExamplePartialInt8s_Add() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) []int8 {
			return v.Add(rosimd.BroadcastInt8(100)).Min(rosimd.BroadcastInt8(102)).Values()
		}),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 101
	// 102
	// 102
}

// Contains is element-wise: it returns a mask marking the lanes that matched, which
// Select then acts on. Reducing a whole stream to one bool is ReduceContainsInt8's job.
func ExamplePartialInt8s_Contains() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](7, 1, 7, 2),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) []int8 {
			matched := v.Contains(rosimd.BroadcastInt8(7))

			return v.Select(matched, rosimd.BroadcastInt8(0)).Values()
		}),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 7
	// 0
	// 7
	// 0
}

// Div exists for the float types only: the standard library has no lane-wise integer
// division.
func ExampleDivFloat64() {
	obs := ro.Pipe3[float64, rosimd.PartialFloat64s, []float64, float64](
		ro.Just[float64](1, 2, 3),
		rosimd.VectorizeFloat64,
		ro.Map(func(v rosimd.PartialFloat64s) []float64 {
			return v.Div(rosimd.BroadcastFloat64(2)).Values()
		}),
		ro.Flatten[float64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value float64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 0.5
	// 1
	// 1.5
}

// The same operator accepts the standard library's own vector type, not just this
// package's Partial types. Only Vectorize and ReduceContains are restricted, because
// they need the validity mask that simd.Int8s does not carry.
func ExampleAddInt8_standardLibraryVector() {
	input := make([]int8, simd.BroadcastInt8s(0).Len())
	for i := range input {
		input[i] = int8(i + 1)
	}

	obs := ro.Pipe1(
		ro.Just(simd.LoadInt8s(input)),
		rosimd.AddInt8(simd.BroadcastInt8s(100)),
	)

	sub := obs.Subscribe(ro.OnNext(func(vector simd.Int8s) {
		var lanes [64]int8
		vector.StorePart(lanes[:])

		// Lane count varies by architecture, so only show the first few.
		fmt.Println(lanes[:3])
	}))
	defer sub.Unsubscribe()

	// Output: [101 102 103]
}
