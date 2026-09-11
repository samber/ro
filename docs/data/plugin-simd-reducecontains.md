---
name: ReduceContains
slug: reducecontains
sourceRef: plugins/exp/simd/contains.go#L36
type: plugin
category: simd
signatures:
  - "func ReduceContainsInt8[V Int8Searchable[V]](target V)"
  - "func ReduceContainsInt16[V Int16Searchable[V]](target V)"
  - "func ReduceContainsInt32[V Int32Searchable[V]](target V)"
  - "func ReduceContainsInt64[V Int64Searchable[V]](target V)"
  - "func ReduceContainsUint8[V Uint8Searchable[V]](target V)"
  - "func ReduceContainsUint16[V Uint16Searchable[V]](target V)"
  - "func ReduceContainsUint32[V Uint32Searchable[V]](target V)"
  - "func ReduceContainsUint64[V Uint64Searchable[V]](target V)"
  - "func ReduceContainsFloat32[V Float32Searchable[V]](target V)"
  - "func ReduceContainsFloat64[V Float64Searchable[V]](target V)"
playUrl:
variantHelpers:
  - plugin#simd#reducecontainsint8
  - plugin#simd#reducecontainsint16
  - plugin#simd#reducecontainsint32
  - plugin#simd#reducecontainsint64
  - plugin#simd#reducecontainsuint8
  - plugin#simd#reducecontainsuint16
  - plugin#simd#reducecontainsuint32
  - plugin#simd#reducecontainsuint64
  - plugin#simd#reducecontainsfloat32
  - plugin#simd#reducecontainsfloat64
similarHelpers:
  - plugin#simd#contains
  - plugin#simd#broadcast
position: 210
---

Reports whether any valid lane of the stream matches `target`.

Widen the value with `Broadcast` so every lane of the target holds it:

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, bool](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(2)),
)

sub := obs.Subscribe(ro.OnNext(func(found bool) {
    fmt.Println(found)
}))
defer sub.Unsubscribe()

// true
```

A value that never appears emits `false` on completion:

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, bool](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.ReduceContainsInt8(rosimd.BroadcastInt8(9)),
)

sub := obs.Subscribe(ro.OnNext(func(found bool) {
    fmt.Println(found)
}))
defer sub.Unsubscribe()

// false
```

It emits as soon as a match is found rather than waiting for completion, so an unbounded stream still produces an answer.

Unlike the arithmetic operators, it accepts only the `Partial` types: answering the question needs the validity mask, since padded lanes are zero-filled and would otherwise report a match on a search for zero.

The element-wise counterpart is the `Contains` method, which returns a per-lane mask rather than a single bool.
