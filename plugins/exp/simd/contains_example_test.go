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
	"math"
	"simd"

	"github.com/samber/ro"
	rosimd "github.com/samber/ro/plugins/exp/simd"
)

// maskLane reports whether one lane of a mask is set.
//
// A mask converts only to the signed vector type of its own width, which is why the
// buffer is int8 even when the mask came from a uint8 vector.
func maskLane(mask simd.Mask8s, lane int) bool {
	vector := mask.ToInt8s()

	var lanes [64]int8
	vector.Store(lanes[:vector.Len()])

	return lanes[lane] != 0
}

// Contains returns a mask rather than a bool, so a caller can read individual lanes.
// Padded lanes are never set: the result is already intersected with the validity mask.
func ExamplePartialInt8s_Contains_readingLanes() {
	// Two valid lanes out of a full register, so every other lane is padding.
	v := rosimd.PartialInt8s{}.LoadPart([]int8{7, 1}, 2)

	matched := v.Contains(rosimd.BroadcastInt8(7))

	fmt.Println(maskLane(matched, 0))
	fmt.Println(maskLane(matched, 1))

	// Output:
	// true
	// false
}

// Contains is element-wise like every other method: it returns a mask — SIMD's vector of
// booleans — marking which lanes matched, already intersected with the validity mask so
// padding is never reported. Select consumes that mask.
//
// Collapsing a whole stream to one answer is ReduceContains' job instead.
func ExamplePartialInt8s_Contains() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](7, 1, 7, 2),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
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

// Masks compose, so two searches can be combined before selecting.
func ExamplePartialInt8s_Contains_combiningMasks() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3, 4),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		ro.Map(func(v rosimd.PartialInt8s) []int8 {
			twos := v.Contains(rosimd.BroadcastInt8(2))
			fours := v.Contains(rosimd.BroadcastInt8(4))

			return v.Select(twos.Or(fours), rosimd.BroadcastInt8(0)).Values()
		}),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 0
	// 2
	// 0
	// 4
}

// Select takes each lane from the receiver where the mask is set and from the other
// vector where it is not.
func ExamplePartialInt8s_Select() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3, 4),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		ro.Map(func(v rosimd.PartialInt8s) []int8 {
			// Keep the lanes below 3, replace the rest with 99.
			small := v.Contains(rosimd.BroadcastInt8(1)).Or(v.Contains(rosimd.BroadcastInt8(2)))

			return v.Select(small, rosimd.BroadcastInt8(99)).Values()
		}),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1
	// 2
	// 99
	// 99
}

// ReduceContains answers the whole-stream question, emitting as soon as a match is found
// so an unbounded stream still produces an answer.
func ExampleReduceContainsInt8() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, bool](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(2)),
	)

	sub := obs.Subscribe(ro.OnNext(func(found bool) {
		fmt.Println(found)
	}))
	defer sub.Unsubscribe()

	// Output: true
}

// A value that never appears emits false on completion.
func ExampleReduceContainsInt8_absent() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, bool](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(9)),
	)

	sub := obs.Subscribe(ro.OnNext(func(found bool) {
		fmt.Println(found)
	}))
	defer sub.Unsubscribe()

	// Output: false
}

// Searching for zero is the case the validity mask exists for. The final batch is
// zero-filled in its padded lanes, which an unmasked compare would report as a match.
func ExampleReduceContainsInt8_zero() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, bool](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(0)),
	)

	sub := obs.Subscribe(ro.OnNext(func(found bool) {
		fmt.Println(found)
	}))
	defer sub.Unsubscribe()

	// Output: false
}

// Searching for NaN never matches, since NaN compares unequal to everything including
// itself — which is what Go's own == does.
func ExampleReduceContainsFloat64_nan() {
	obs := ro.Pipe2[float64, rosimd.PartialFloat64s, bool](
		ro.Just(1.0, math.NaN(), 3.0),
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.ReduceContainsFloat64(rosimd.BroadcastFloat64(math.NaN())),
	)

	sub := obs.Subscribe(ro.OnNext(func(found bool) {
		fmt.Println(found)
	}))
	defer sub.Unsubscribe()

	// Output: false
}
