# exp/simd — SIMD vector operators for ro

SIMD operators for [ro](../../), built on the portable `simd` package introduced in Go 1.27. Portable means one implementation compiles for amd64, arm64 and wasm, with a pure-Go fallback — there are no per-ISA kernels here.

> **Experimental.** This package tracks a Go experiment. Its API may change with the toolchain, and it is excluded from the workspace build.

## Requirements

- Go 1.27 or later
- `GOEXPERIMENT=simd`

The plugin is deliberately left out of `go.work`, so build and test it on its own:

```bash
cd plugins/exp/simd
GOWORK=off GOEXPERIMENT=simd go test ./...
```

## The idea

A stream carries one value at a time; SIMD works on a whole register at once. `VectorizeInt8` bridges the two by batching scalars into vectors.

Streams rarely deliver a multiple of the lane width, so the last vector of a batch is short. That is the problem `PartialInt8s` solves: it carries a validity mask alongside its lanes, so a short vector is an ordinary value rather than a special case. Every operation leaves padded lanes at their previous value, and nothing downstream can observe them.

```go
import (
    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

sum, _ := ro.Collect(
    ro.Pipe3[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, int8](
        ro.FromSlice([]int8{1, 2, 3, 4, 5}),
        rosimd.VectorizeInt8,
        rosimd.AddInt8(rosimd.BroadcastInt8(10)),
        rosimd.ReduceSumInt8,
    ),
)
// [65]
```

## Operands are vectors, not scalars

SIMD has no scalar-operand arithmetic — `Add` takes another vector — and a generic operator cannot widen a scalar for itself without breaking the compiler's specialization pass. Widen constants at the call site:

```go
rosimd.AddInt8(rosimd.BroadcastInt8(42))
```

This is also what lets the type argument be inferred, so call sites stay free of `[rosimd.PartialInt8s]`.

## Works with standard library vectors too

Operators are generic over an interface satisfied by both `rosimd.PartialInt8s` and the standard library's own `simd.Int8s`:

```go
ro.Pipe1(vectorStream, rosimd.AddInt8(simd.BroadcastInt8s(42)))
```

Operators that need the validity mask — `VectorizeInt8`, `ReduceContainsInt8` — accept only the `Partial` types, because `simd.Int8s` carries no mask.

## Methods or operators

Element-wise work is available both ways. Methods chain inside `ro.Map` and need no type arguments, which is usually shorter:

```go
ro.Map(func(v rosimd.PartialInt8s) []int8 {
    return v.Add(rosimd.BroadcastInt8(42)).Min(rosimd.BroadcastInt8(50)).Values()
})
```

## Comparing lanes

`Contains` is element-wise, like every other method: it returns a **mask** — SIMD's vector of booleans — saying which lanes matched, already intersected with the validity mask so padding is never reported. `Select` consumes that mask:

```go
ro.Map(func(v rosimd.PartialInt8s) []int8 {
    matched := v.Contains(rosimd.BroadcastInt8(7))

    return v.Select(matched, rosimd.BroadcastInt8(0)).Values()
})
// lanes equal to 7 keep their value, every other lane becomes 0
```

To collapse a whole stream to a single answer instead, use the `ReduceContains` operator.

## Leaving vector space

There is no devectorize operator. Use `ro.Map` plus `ro.Flatten`, or one of the `Reduce` operators:

```go
ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
    source,
    rosimd.VectorizeInt8,
    ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
    ro.Flatten[int8](),
)
```

## Operator coverage

The standard library's vector types are not uniform, so neither is this package. An operator exists only where the underlying operation does.

Ten element types are covered: `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Float32`, `Float64`.

| Operation                                               | Available for                                                                             |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `Vectorize`, `Broadcast`                                | every element type                                                                        |
| `Add`, `Sub`, `Min`, `Max`, `Clamp`                     | every element type                                                                        |
| `Contains`, `Select` (methods)                          | every `Partial` type — `Contains` returns a mask, `Select` consumes one                   |
| `Mul`                                                   | every type except `Int64` and `Uint64` — the standard library has no 64-bit lane multiply |
| `Div`                                                   | `Float32` and `Float64` only                                                              |
| `ReduceSum`, `ReduceMin`, `ReduceMax`, `ReduceContains` | every element type                                                                        |

`Add`, `Sub`, `Mul`, `Div`, `Min` and `Max` each have a `With` variant that combines two vector streams in lockstep, following `ro.ZipWith`'s naming — `AddWithInt8`, `MinWithFloat64`, and so on. `Clamp` has none: it takes two bounds, so a two-stream form would have to zip three streams at once.

`Int64` and `Uint64` accept only the `Partial` types. `simd.Int64s` and `simd.Uint64s` have no `Min` or `Max`, so the `Partial` types synthesize them from `Less` and `IfElse` — which is precisely what the wrapper is for, since the standard library's per-type method sets are not uniform. Every other element type accepts stdlib vectors too.

### NaN

For `Float32` and `Float64`, element-wise `Min` and `Max` are architecture-dependent in hardware: x86 discards NaN, arm64 propagates it. The `Partial` types detect NaN lanes explicitly and force it into the result, so behaviour is identical everywhere and matches Go's own `min`/`max` builtins. Passing a stdlib `simd.Float64s` through the same operator uses the raw instruction and does **not** carry that guarantee.

Reductions deliberately behave the other way. `ReduceMin` and `ReduceMax` compare with `<` and `>`, both false for NaN, so a NaN never displaces the accumulator — matching core `ro.Min` and `ro.Max` exactly rather than Go's NaN-propagating builtins.

## Performance

Vectorizing does not automatically make a `ro` pipeline faster. Measurements on the previous implementation showed the reactive machinery — per-item dispatch, context propagation, channel handoff — dominating the arithmetic at every input size tested. Batching amortises that cost, which is the point of this design, but benchmark your own pipeline rather than assuming a win.

## Package layout

Operators are grouped by what they do, across all ten element types. The per-type files hold what genuinely varies per type, which is also the only code that touches `simd` directly.

| File                     | Holds                                                                              |
| ------------------------ | ---------------------------------------------------------------------------------- |
| `vectorize.go`           | `Vectorize` — the entry point into vector space                                    |
| `arithmetic.go`          | `Add`, `Sub`, `Mul`, `Div` and their `With` variants                               |
| `bounds.go`              | `Min`, `Max`, `Clamp`, `MinWith`, `MaxWith`                                        |
| `contains.go`            | `ReduceContains`                                                                   |
| `reduce.go`              | `ReduceSum`, `ReduceMin`, `ReduceMax`                                              |
| `vector.go`              | the generic plumbing those share                                                   |
| `int8.go` … `float64.go` | per type: constraint interfaces, the `Partial` struct, `Broadcast`, masks, methods |

Tests and examples mirror the same split, with the per-type test fixtures in `testhelpers_test.go`.

## Contributing

The Go 1.27 compiler rewrites every function that touches a simd type into a dispatcher plus per-width clones, and several ordinary-looking Go constructs do not survive that rewrite. The "Editing this package" section of the [package doc](https://pkg.go.dev/github.com/samber/ro/plugins/exp/simd) states the rules; [COMPILER-CONSTRAINTS.md](./COMPILER-CONSTRAINTS.md) records the probes that established them, including the exact error each rejected shape produces. Read the latter before concluding a rule is wrong — several shapes that look obviously fine do not compile.

The short version:

1. No function may put a concrete simd-containing type inside another package's generic type in its own signature — `ro.Observable[PartialInt8s]` fails as a return type and as a callback parameter alike. Naming a bare simd type is fine.
2. The concrete type belongs at the call site, as an explicit type argument, never inside such a declaration.
3. `simd.*` calls and struct-literal construction live in methods on the concrete type, never inside a generic function's own body.
4. Prefer stage functions that take the source directly over curried ones — only the former let Go infer the type argument.
5. Every file with simd-dependent code must import `simd` and touch it in a function body. A file counts as simd-dependent when it names a concrete `Partial` type — the operator files are exempt despite driving all the vector work, because they stay generic throughout.
