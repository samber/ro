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

// Package rosimd adds SIMD vector operators to samber/ro, built on the portable
// simd package introduced in Go 1.27.
//
// A stream carries one value at a time, while SIMD works on a whole register at
// once. VectorizeInt8 bridges the two by batching scalars into vectors, and because
// a stream rarely delivers a multiple of the lane width, the trailing vector is
// short. PartialInt8s carries a validity mask alongside its lanes so that short
// vector is an ordinary value rather than a special case: every operation leaves
// padded lanes at their previous value, and nothing downstream can observe them.
//
//	ro.Pipe3[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.AddInt8(rosimd.BroadcastInt8(42)),
//		rosimd.ReduceSumInt8[rosimd.PartialInt8s](),
//	)
//
// Operators accept either PartialInt8s or the standard library's own simd.Int8s,
// because they are generic over an interface both satisfy. Operands are vectors
// rather than scalars — SIMD has no scalar-operand arithmetic — so widen constants
// with BroadcastInt8 at the call site. ReduceContains is the one that takes a scalar
// directly, as in ReduceContainsInt8[PartialInt8s](7): a search target is a single
// value, not an operand.
//
// # Element types
//
// Ten types are covered: Int8, Int16, Int32, Int64, Uint8, Uint16, Uint32, Uint64,
// Float32 and Float64, each with its own Partial type and its own suffixed operators.
//
// The standard library's method sets are not uniform, so neither is the coverage.
// Mul exists for every type except Int64 and Uint64, which have no 64-bit lane
// multiply to build on. Div exists only for the float types. Int64 and Uint64 accept
// only their Partial types, because simd.Int64s and simd.Uint64s have no Min or Max
// and the Partial types synthesize those from Less and IfElse.
//
// # NaN
//
// For the float types, element-wise Min and Max are architecture-dependent in
// hardware: x86 discards NaN, arm64 propagates it. The Partial types detect NaN lanes
// and force it into the result, so behaviour is identical everywhere and matches Go's
// min and max builtins. A stdlib simd.Float64s passed through the same operator uses
// the raw instruction and does not carry that guarantee.
//
// Reductions deliberately differ: ReduceMin and ReduceMax compare with < and >, both
// false for NaN, so a NaN never displaces the accumulator. That matches core ro.Min
// and ro.Max rather than the NaN-propagating builtins.
//
// Operators chain in a Pipe like any other ro operator, each stage keeping the stream in
// vector space:
//
//	ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, rosimd.PartialInt8s, int8](
//		source,
//		rosimd.VectorizeInt8[rosimd.PartialInt8s](),
//		rosimd.AddInt8(rosimd.BroadcastInt8(42)),
//		rosimd.MinInt8(rosimd.BroadcastInt8(50)),
//		rosimd.Flatten[rosimd.PartialInt8s](),
//	)
//
// The same operations exist as methods on the Partial types. That is how the operators
// reach them, through the constraint interface, and it is what lets simd.Int8s satisfy
// the same interface — so the methods are the mechanism rather than the usual way to
// call one. Reach for them when there is no operator, as with Contains and Select.
//
// Contains is element-wise like the rest: it returns a mask — SIMD's vector of
// booleans — marking which lanes matched, already intersected with the validity mask
// so padding is never reported. Select consumes that mask.
//
// A mask is not a vector, so no operator can carry one between stages: a search and the
// select it feeds belong together in a single ro.Map, which keeps the stream in vector
// space for the operators on either side.
//
//	ro.Map(func(v rosimd.PartialInt8s) rosimd.PartialInt8s {
//		matched := v.Contains(rosimd.BroadcastInt8(7))
//
//		return v.Select(matched, rosimd.BroadcastInt8(0))
//	})
//
// Collapsing a whole stream to a single answer is the ReduceContains operator's job
// instead.
//
// Leaving vector space is ToScalar, which hands each vector's valid lanes back as a
// slice, or Flatten, which emits them one at a time — ToScalar followed by ro.Flatten in
// a single stage.
//
// The Reduce operators are the other way out, and the only ones that accumulate:
// everything else is a single vectorized streamed operation, a vector in and a vector
// out, with nothing carried between items. A Reduce is an Unvectorize, the opposite of
// Vectorize — it accumulates across vectors and also horizontally within each vector,
// and collapses the whole stream to one value on completion.
//
// Count is the operator for when the batching itself is the question rather than the
// data: one lane count per vector, short only for a stream's final batch.
//
// Both exits ask only that a vector can report its lanes, so unlike the arithmetic
// operators they accept simd.Int64s and simd.Uint64s too. Both are also unsuffixed,
// alone among the operators here: only the vector type is written at the call site, as in
// Flatten[PartialInt8s](), and the element type is inferred from that vector's own
// StorePart signature — so one operator serves all ten types.
//
// # Package layout
//
// The operators are grouped by what they do, across all ten element types:
// vectorize.go, arithmetic.go, bounds.go, contains.go and reduce.go. vector.go holds the
// generic plumbing they share.
//
// The per-type files — int8.go through float64.go — hold what genuinely varies per type,
// which is also the only code that touches simd directly: the constraint interfaces, the
// Partial struct, Broadcast, the mask helpers and every method.
//
// # Build requirements
//
// This package requires GOEXPERIMENT=simd, and is excluded from the workspace:
//
//	cd plugins/exp/simd && GOWORK=off GOEXPERIMENT=simd go test ./...
//
// # Performance
//
// Vectorizing does not automatically make a ro pipeline faster. Measurements on the
// previous implementation showed the reactive machinery — per-item dispatch, context
// propagation, channel handoff — dominating the arithmetic at every input size.
// Batching amortises that cost, but benchmark before assuming a win.
//
// # Editing this package
//
// The Go 1.27 compiler rewrites every function that touches a simd type into a
// dispatcher plus per-width clones, and several ordinary-looking Go constructs do
// not survive that rewrite:
//
//  1. No function may put a concrete simd-containing type inside another package's
//     generic type in its own signature — ro.Observable[PartialInt8s] fails both as
//     a return type and as a callback parameter. Keep those declarations generic
//     over the vector type parameter. Naming a bare simd type is fine, which is why
//     fullMaskInt8() simd.Mask8s compiles.
//  2. The concrete type belongs at the call site, as an explicit type argument,
//     never inside such a declaration.
//  3. simd.* calls and struct-literal construction live in methods on the concrete
//     type, reached only through the constraint interface — never inside a generic
//     function's own body, even indirectly through a plain helper function.
//  4. A curried operator's vector type must be written at the call site. Its type
//     parameter appears only in the type of the func it returns, which Go's inference
//     does not reach, so Vectorize, ToScalar and Flatten are always instantiated
//     explicitly. The reductions take the source directly instead, letting the
//     surrounding Pipe pin their type argument.
//  5. Every file with simd-dependent code must import "simd" and touch it inside a
//     function body; a package-level var reference does not satisfy the specializer. A
//     file counts as simd-dependent when it names a concrete Partial type. The operator
//     files are exempt even though they drive all the vector work, because they stay
//     generic throughout and reach simd only through methods on their type parameter.
//
// COMPILER-CONSTRAINTS.md records the probes behind these rules and the exact error
// each rejected shape produces; read it before concluding a rule is wrong, because
// several shapes that look obviously fine do not compile.
package rosimd
