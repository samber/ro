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

// Vectorize batches a scalar stream into vectors. Leaving vector space again is ro.Map
// plus ro.Flatten, or one of the Reduce operators — there is no devectorize.
func ExampleVectorizeInt8() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3, 4, 5),
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
	// 4
	// 5
}

// A stream rarely delivers a multiple of the lane width, so the final batch is short. It
// is emitted as an ordinary vector whose padded lanes are masked off, rather than being
// dropped or padded into the result.
//
// The lane count varies by architecture, so this compares against it rather than
// printing it.
func ExampleVectorizeInt8_shortFinalBatch() {
	lanes := simd.BroadcastInt8s(0).Len()

	// One and a half registers' worth, so the second batch is half full.
	input := make([]int8, lanes+lanes/2)
	for i := range input {
		input[i] = int8(i%100 + 1)
	}

	counts, err := ro.Collect(
		ro.Pipe2[int8, rosimd.PartialInt8s, int](
			ro.FromSlice(input),
			rosimd.VectorizeInt8,
			ro.Map(func(v rosimd.PartialInt8s) int { return v.Count() }),
		),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(len(counts))
	fmt.Println(counts[0] == lanes)
	fmt.Println(counts[1] == lanes/2)

	// Output:
	// 2
	// true
	// true
}

// There is one Vectorize per element type.
func ExampleVectorizeFloat64() {
	obs := ro.Pipe3[float64, rosimd.PartialFloat64s, []float64, float64](
		ro.Just(1.5, 2.5, 3.5),
		rosimd.VectorizeFloat64,
		ro.Map(func(v rosimd.PartialFloat64s) []float64 { return v.Values() }),
		ro.Flatten[float64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value float64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1.5
	// 2.5
	// 3.5
}

// An empty source emits no vector at all, rather than one made entirely of padding.
func ExampleVectorizeInt8_empty() {
	vectors, err := ro.Collect(rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Empty[int8]()))
	if err != nil {
		panic(err)
	}

	fmt.Println(len(vectors))

	// Output: 0
}
