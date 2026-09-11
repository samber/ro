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

// Every element-wise operator is also a method. Methods chain inside ro.Map and need no
// type arguments, which is usually shorter than stacking operators.
func ExamplePartialInt8s() {
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

// Add leaves padded lanes at their previous value, so a short final batch emits exactly
// the values it holds and no more.
func ExamplePartialInt8s_Add() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) []int8 {
			return v.Add(rosimd.BroadcastInt8(10)).Values()
		}),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 11
	// 12
	// 13
}

// Values returns the valid lanes as a slice, which is how a pipeline leaves vector
// space. Pair it with ro.Flatten to get a scalar stream back.
func ExamplePartialInt8s_Values() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, []int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
	)

	sub := obs.Subscribe(ro.OnNext(func(values []int8) {
		fmt.Println(values)
	}))
	defer sub.Unsubscribe()

	// Output: [1 2 3]
}

// Count reports how many lanes hold data, which for a short final batch is fewer than
// the lane capacity.
func ExamplePartialInt8s_Count() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) int { return v.Count() }),
	)

	sub := obs.Subscribe(ro.OnNext(func(count int) {
		fmt.Println(count)
	}))
	defer sub.Unsubscribe()

	// Output: 3
}

// Sum folds the valid lanes of one vector to a scalar. To total a whole stream instead,
// use the ReduceSum operator.
func ExamplePartialInt8s_Sum() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8,
		ro.Map(func(v rosimd.PartialInt8s) int8 { return v.Sum() }),
	)

	sub := obs.Subscribe(ro.OnNext(func(total int8) {
		fmt.Println(total)
	}))
	defer sub.Unsubscribe()

	// Output: 6
}

// Len is the lane capacity of the running architecture, not the valid count, and is
// meaningful on the zero value — which is how Vectorize discovers its batch size.
func ExamplePartialInt8s_Len() {
	var empty rosimd.PartialInt8s

	fmt.Println(empty.Len() == simd.BroadcastInt8s(0).Len())
	fmt.Println(empty.Count())

	// Output:
	// true
	// 0
}
