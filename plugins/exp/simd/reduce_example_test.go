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

// The vector type is given at the call site, since currying puts it out of inference's
// reach.
func ExampleReduceSumInt8() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Just[int8](1, 2, 3, 4, 5),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceSumInt8[rosimd.PartialInt8s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(total int8) {
		fmt.Println(total)
	}))
	defer sub.Unsubscribe()

	// Output: 15
}

// The sum accumulates in the element type and wraps on overflow, exactly as ro.Sum does.
// An empty stream sums to zero.
func ExampleReduceSumInt8_empty() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Empty[int8](),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceSumInt8[rosimd.PartialInt8s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(total int8) {
		fmt.Println(total)
	}))
	defer sub.Unsubscribe()

	// Output: 0
}

func ExampleReduceMinInt8() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Just[int8](5, 2, 8),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceMinInt8[rosimd.PartialInt8s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(smallest int8) {
		fmt.Println(smallest)
	}))
	defer sub.Unsubscribe()

	// Output: 2
}

func ExampleReduceMaxInt8() {
	obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
		ro.Just[int8](5, 2, 8),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ReduceMaxInt8[rosimd.PartialInt8s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(largest int8) {
		fmt.Println(largest)
	}))
	defer sub.Unsubscribe()

	// Output: 8
}

// An empty stream emits nothing at all, rather than a zero that would be
// indistinguishable from a real result. This matches ro.Min.
func ExampleReduceMinInt8_empty() {
	values, err := ro.Collect(
		ro.Pipe2[int8, rosimd.PartialInt8s, int8](
			ro.Empty[int8](),
			rosimd.VectorizeInt8[rosimd.PartialInt8s](),
			rosimd.ReduceMinInt8[rosimd.PartialInt8s](),
		),
	)
	if err != nil {
		panic(err)
	}

	fmt.Println(len(values))

	// Output: 0
}

// The reductions compare with < and >, both false for NaN, so a NaN never displaces the
// accumulator — matching core ro.Min and ro.Max. The element-wise MinFloat64 does the
// opposite and propagates it.
func ExampleReduceMinFloat64_nan() {
	obs := ro.Pipe2[float64, rosimd.PartialFloat64s, float64](
		ro.Just(3.0, math.NaN(), 1.0),
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.ReduceMinFloat64[rosimd.PartialFloat64s](),
	)

	sub := obs.Subscribe(ro.OnNext(func(smallest float64) {
		fmt.Println(smallest)
	}))
	defer sub.Unsubscribe()

	// Output: 1
}

// The reductions accept the standard library's vector type too. Such a vector carries no
// validity mask, so every lane counts.
func ExampleReduceSumInt8_standardLibraryVector() {
	input := make([]int8, simd.BroadcastInt8s(0).Len())
	for i := range input {
		input[i] = 1
	}

	obs := rosimd.ReduceSumInt8[simd.Int8s]()(ro.Just(simd.LoadInt8s(input)))

	sub := obs.Subscribe(ro.OnNext(func(total int8) {
		// The lane count varies by architecture, so compare rather than print.
		fmt.Println(int(total) == len(input))
	}))
	defer sub.Unsubscribe()

	// Output: true
}
