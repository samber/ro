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

// Vectorize batches a scalar stream into vectors. ToScalar brings it back out as one
// slice per vector, which ro.Flatten then unpacks into values.
func ExampleVectorizeInt8() {
	obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3, 4, 5),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ToScalar[rosimd.PartialInt8s](),
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
// dropped or padded into the result. Count reports that shape.
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
			rosimd.VectorizeInt8[rosimd.PartialInt8s](),
			rosimd.Count[rosimd.PartialInt8s](),
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
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.ToScalar[rosimd.PartialFloat64s](),
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

// ToScalar hands each vector's valid lanes back as a slice, one slice per vector. It is
// the exit from vector space, and the shape is the batching: a short final batch gives a
// correspondingly short slice.
func ExampleToScalar() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, []int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ToScalar[rosimd.PartialInt8s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(lanes []int8) {
		fmt.Println(lanes)
	}))
	defer sub.Unsubscribe()

	// Output: [1 2 3]
}

// Flatten is ToScalar followed by ro.Flatten in a single stage: one value per lane
// instead of one slice per vector. It turns a vector stream straight back into the
// scalar stream Vectorize was given.
func ExampleFlatten() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Just[int8](1, 2, 3, 4, 5),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.Flatten[rosimd.PartialInt8s](),
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

// Neither exit needs arithmetic, only the ability to report lanes, so both accept the
// standard library's vector types — simd.Int64s included, which the arithmetic operators
// reject for want of Min and Max.
func ExampleFlatten_standardLibraryVector() {
	input := make([]int8, simd.BroadcastInt8s(0).Len())
	for i := range input {
		input[i] = int8(i + 1)
	}

	values, err := ro.Collect(rosimd.Flatten[simd.Int8s]()(ro.Just(simd.LoadInt8s(input))))
	if err != nil {
		panic(err)
	}

	// Lane count varies by architecture, so only the first few are shown.
	fmt.Println(values[:3])

	// Output: [1 2 3]
}

// An empty source emits no vector at all, rather than one made entirely of padding.
func ExampleVectorizeInt8_empty() {
	vectors, err := ro.Collect(rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Empty[int8]()))
	if err != nil {
		panic(err)
	}

	fmt.Println(len(vectors))

	// Output: 0
}
