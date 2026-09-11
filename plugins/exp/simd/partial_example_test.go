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

// PartialInt8s carries a validity mask alongside its lanes, so the short final batch of a
// stream is an ordinary value rather than a special case. Every operation leaves padded
// lanes at their previous value, and nothing downstream can observe them.
//
// Operators chain in a Pipe like any other ro operator, each stage keeping the stream in
// vector space until ToScalar takes it back out.
func ExamplePartialInt8s() {
	obs := ro.Pipe5[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.AddInt8(rosimd.BroadcastInt8(100)),
		rosimd.MinInt8(rosimd.BroadcastInt8(102)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
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

// Values returns the valid lanes as a slice, padding excluded. The ToScalar operator does
// this for a whole stream, so the method is for reading a single vector directly — the
// result of a Select, say.
func ExamplePartialInt8s_Values() {
	v := rosimd.PartialInt8s{}.LoadPart([]int8{1, 2, 3}, 3)

	fmt.Println(v.Values())
	fmt.Println(v.Select(v.Contains(rosimd.BroadcastInt8(2)), rosimd.BroadcastInt8(0)).Values())

	// Output:
	// [1 2 3]
	// [0 2 0]
}

// Count reports how many lanes hold data, which for a short batch is fewer than the lane
// capacity Len reports. The Count operator applies it to a whole stream.
func ExamplePartialInt8s_Count() {
	v := rosimd.PartialInt8s{}.LoadPart([]int8{1, 2, 3}, 3)

	fmt.Println(v.Count())

	// The lane count varies by architecture, but never dips below 16 for int8.
	fmt.Println(v.Count() < v.Len())

	// Output:
	// 3
	// true
}

// Sum folds the valid lanes of one vector to a scalar, ignoring padding. To total a whole
// stream instead, use the ReduceSum operator.
func ExamplePartialInt8s_Sum() {
	v := rosimd.PartialInt8s{}.LoadPart([]int8{1, 2, 3}, 3)

	fmt.Println(v.Sum())

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
