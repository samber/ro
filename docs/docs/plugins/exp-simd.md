---
title: SIMD (experimental)
description: SIMD operators for ro — Go reactive streams. Batch a stream into vectors and run lane-wise math on amd64, arm64 and wasm with Go 1.27+ and GOEXPERIMENT=simd.
sidebar_position: 300
hide_table_of_contents: true
---

# SIMD - Plugin operators

This page lists all operators available in the `exp/simd` sub-package. They are built on the portable `simd` package introduced in Go 1.27: one implementation compiles for amd64, arm64 and wasm, with a pure-Go fallback, so there are no per-ISA operators to choose between. The register width is discovered at runtime.

A stream carries one value at a time; SIMD works on a whole register at once. `Vectorize` bridges the two by batching scalars into vectors, and `Partial` types carry a validity mask so the short final vector of a batch is an ordinary value rather than a special case. `ToScalar` and `Flatten` bring the stream back out — as one slice per vector, or one value per lane — and the `Reduce` operators collapse it to a single value instead.

```go
ro.Pipe3[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, int8](
    ro.FromSlice([]int8{1, 2, 3, 4, 5}),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.AddInt8(rosimd.BroadcastInt8(10)),
    rosimd.ReduceSumInt8[rosimd.PartialInt8s](),
)
// 65
```

Ten element types are covered — `Int8`, `Int16`, `Int32`, `Int64`, `Uint8`, `Uint16`, `Uint32`, `Uint64`, `Float32`, `Float64` — but the standard library's vector types are not uniform, so neither is the coverage: `Mul` is missing for the 64-bit integers, and `Div` exists for the float types only.

:::warning Help improve this documentation
This documentation is still new and evolving. If you spot any mistakes, unclear explanations, or missing details, please [open an issue](https://github.com/samber/ro/issues).

Your feedback helps us improve!
:::

:::warning Unstable API
SIMD operators are experimental. The API may break in the future.
:::

### Install

First, import the sub-package in your project:

```bash
go get -u github.com/samber/ro/plugins/exp/simd
```

The plugin needs Go 1.27 or later and `GOEXPERIMENT=simd`. It is deliberately left out of the workspace, so build and test it on its own:

```bash
GOWORK=off GOEXPERIMENT=simd go test ./...
```

import HelperList from '@site/plugins/helpers-pages/components/HelperList';

<HelperList
  type="plugin"
  category="simd"
/>
