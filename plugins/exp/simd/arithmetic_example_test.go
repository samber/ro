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

// The operand is a vector, not a scalar: SIMD has no scalar-operand arithmetic. Widen
// constants at the call site with Broadcast, which is also what lets the type argument
// be inferred.
func ExampleAddInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.AddInt8(rosimd.BroadcastInt8(10)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
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

// Operators chain in a Pipe like any other ro operator. Each stage keeps the stream in
// vector space, so the arithmetic happens lane-wise the whole way down, until ToScalar
// takes it back out.
func ExampleAddInt8_chained() {
	obs := ro.Pipe5[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.AddInt8(rosimd.BroadcastInt8(10)),
		rosimd.MulInt8(rosimd.BroadcastInt8(2)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 22
	// 24
	// 26
}

func ExampleSubInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](10, 20, 30),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.SubInt8(rosimd.BroadcastInt8(5)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 5
	// 15
	// 25
}

// Mul exists for every element type except Int64 and Uint64, which have no 64-bit lane
// multiply in the standard library.
func ExampleMulInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 2, 3),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.MulInt8(rosimd.BroadcastInt8(3)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 3
	// 6
	// 9
}

// Div exists for the float types only: the standard library provides no lane-wise
// integer division.
func ExampleDivFloat64() {
	obs := ro.Pipe4[float64, rosimd.PartialFloat64s, rosimd.PartialFloat64s, []float64, float64](
		ro.Just[float64](1, 2, 3),
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.DivFloat64(rosimd.BroadcastFloat64(2)),
		rosimd.ToScalar[rosimd.PartialFloat64s](),
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

// Dividing by zero yields ±Inf, and 0/0 yields NaN, exactly as Go's own float division
// does. Neither is an error, and neither is suppressed.
func ExampleDivFloat64_byZero() {
	obs := ro.Pipe4[float64, rosimd.PartialFloat64s, rosimd.PartialFloat64s, []float64, float64](
		ro.Just[float64](1, -1, 0),
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.DivFloat64(rosimd.BroadcastFloat64(0)),
		rosimd.ToScalar[rosimd.PartialFloat64s](),
		ro.Flatten[float64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value float64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// +Inf
	// -Inf
	// NaN
}

// Arithmetic wraps in the element type, exactly as Go's own operators do. The standard
// library offers saturating alternatives; this package deliberately does not use them.
func ExampleAddUint8_overflow() {
	obs := ro.Pipe4[uint8, rosimd.PartialUint8s, rosimd.PartialUint8s, []uint8, uint8](
		ro.Just[uint8](250, 10),
		rosimd.VectorizeUint8[rosimd.PartialUint8s](),
		rosimd.AddUint8(rosimd.BroadcastUint8(10)),
		rosimd.ToScalar[rosimd.PartialUint8s](),
		ro.Flatten[uint8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value uint8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 4
	// 20
}

// The With variants combine two vector streams in lockstep instead of a constant
// operand, following ro.ZipWith's naming.
func ExampleAddInt8With() {
	left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 2, 3))
	right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 20, 30))

	obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
		rosimd.AddInt8With(right)(left),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 11
	// 22
	// 33
}

func ExampleDivFloat64With() {
	left := rosimd.VectorizeFloat64[rosimd.PartialFloat64s]()(ro.Just[float64](10, 20, 30))
	right := rosimd.VectorizeFloat64[rosimd.PartialFloat64s]()(ro.Just[float64](2, 4, 5))

	obs := ro.Pipe2[rosimd.PartialFloat64s, []float64, float64](
		rosimd.DivFloat64With(right)(left),
		rosimd.ToScalar[rosimd.PartialFloat64s](),
		ro.Flatten[float64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value float64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 5
	// 5
	// 6
}

// The operators are generic over an interface that the standard library's own vector
// types satisfy too, so a stream of simd.Int8s needs no wrapping. Only Vectorize and
// ReduceContains are restricted to the Partial types, because they need the validity
// mask that simd.Int8s does not carry.
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

		// Lane count varies by architecture, so only the first few are shown.
		fmt.Println(lanes[:3])
	}))
	defer sub.Unsubscribe()

	// Output: [101 102 103]
}
