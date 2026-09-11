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

// Min is element-wise, one vector out per vector in. ReduceMinInt8 is the aggregating
// counterpart that collapses a whole stream to one value.
func ExampleMinInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 50, 100),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.MinInt8(rosimd.BroadcastInt8(60)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1
	// 50
	// 60
}

func ExampleMaxInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 50, 100),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.MaxInt8(rosimd.BroadcastInt8(40)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 40
	// 50
	// 100
}

// Clamp composes Max(lower) then Min(upper). Passing lower greater than upper is a
// programmer error that this cannot detect — the bounds are opaque vectors — and
// collapses every lane to upper, where core ro.Clamp would panic at construction.
func ExampleClampInt8() {
	obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
		ro.Just[int8](1, 50, 100),
		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
		rosimd.ClampInt8(rosimd.BroadcastInt8(10), rosimd.BroadcastInt8(60)),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 10
	// 50
	// 60
}

// MinWith takes the smaller of each lane pair from two streams rather than from a
// constant.
func ExampleMinInt8With() {
	left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 50, 100))
	right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 10, 10))

	obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
		rosimd.MinInt8With(right)(left),
		rosimd.ToScalar[rosimd.PartialInt8s](),
		ro.Flatten[int8](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value int8) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1
	// 10
	// 10
}

// Element-wise Min and Max force NaN into the result, matching Go's own min and max
// builtins. The hardware disagrees about this — x86 discards NaN, arm64 propagates it —
// so the Partial types detect NaN lanes and settle it, making every architecture agree.
//
// The reductions deliberately do the opposite: see ReduceMinFloat64.
func ExampleMinFloat64_nan() {
	obs := ro.Pipe4[float64, rosimd.PartialFloat64s, rosimd.PartialFloat64s, []float64, float64](
		ro.Just(1.0, math.NaN(), 3.0),
		rosimd.VectorizeFloat64[rosimd.PartialFloat64s](),
		rosimd.MinFloat64(rosimd.BroadcastFloat64(2)),
		rosimd.ToScalar[rosimd.PartialFloat64s](),
		ro.Flatten[float64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value float64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 1
	// NaN
	// 2
}

// simd.Uint64s has no Min instruction, so PartialUint64s synthesizes one from Less and
// IfElse. The comparison stays unsigned: a value above the signed maximum is large, not
// negative.
//
// This is also why Uint64 and Int64 accept only the Partial types — the standard
// library's own 64-bit vectors cannot satisfy the constraint.
func ExampleMinUint64() {
	obs := ro.Pipe4[uint64, rosimd.PartialUint64s, rosimd.PartialUint64s, []uint64, uint64](
		ro.Just[uint64](100, math.MaxUint64),
		rosimd.VectorizeUint64[rosimd.PartialUint64s](),
		rosimd.MinUint64(rosimd.BroadcastUint64(500)),
		rosimd.ToScalar[rosimd.PartialUint64s](),
		ro.Flatten[uint64](),
	)

	sub := obs.Subscribe(ro.OnNext(func(value uint64) {
		fmt.Println(value)
	}))
	defer sub.Unsubscribe()

	// Output:
	// 100
	// 500
}

// Like the arithmetic operators, Min accepts the standard library's vector type.
func ExampleMinInt8_standardLibraryVector() {
	input := make([]int8, simd.BroadcastInt8s(0).Len())
	for i := range input {
		input[i] = int8(i + 1)
	}

	obs := ro.Pipe1(
		ro.Just(simd.LoadInt8s(input)),
		rosimd.MinInt8(simd.BroadcastInt8s(2)),
	)

	sub := obs.Subscribe(ro.OnNext(func(vector simd.Int8s) {
		var lanes [64]int8
		vector.StorePart(lanes[:])

		// Lane count varies by architecture, so only the first few are shown.
		fmt.Println(lanes[:3])
	}))
	defer sub.Unsubscribe()

	// Output: [1 2 2]
}
