---
name: Clamp
slug: clamp
sourceRef: plugins/exp/simd/bounds.go#L231
type: plugin
category: simd
signatures:
  - "func ClampInt8[V Int8Vector[V]](lower, upper V)"
  - "func ClampInt16[V Int16Vector[V]](lower, upper V)"
  - "func ClampInt32[V Int32Vector[V]](lower, upper V)"
  - "func ClampInt64[V Int64Vector[V]](lower, upper V)"
  - "func ClampUint8[V Uint8Vector[V]](lower, upper V)"
  - "func ClampUint16[V Uint16Vector[V]](lower, upper V)"
  - "func ClampUint32[V Uint32Vector[V]](lower, upper V)"
  - "func ClampUint64[V Uint64Vector[V]](lower, upper V)"
  - "func ClampFloat32[V Float32Vector[V]](lower, upper V)"
  - "func ClampFloat64[V Float64Vector[V]](lower, upper V)"
playUrl:
variantHelpers:
  - plugin#simd#clampint8
  - plugin#simd#clampint16
  - plugin#simd#clampint32
  - plugin#simd#clampint64
  - plugin#simd#clampuint8
  - plugin#simd#clampuint16
  - plugin#simd#clampuint32
  - plugin#simd#clampuint64
  - plugin#simd#clampfloat32
  - plugin#simd#clampfloat64
similarHelpers:
  - plugin#simd#min
  - plugin#simd#max
position: 110
---

Bounds every lane to `[lower, upper]`.

The operand is a vector, not a scalar. SIMD has no scalar-operand arithmetic, so widen the value at the call site with the matching `Broadcast` — which is also what lets the type argument be inferred, keeping the call free of an explicit `[rosimd.PartialInt8s]`.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](1, 50, 100),
    rosimd.VectorizeInt8,
    rosimd.ClampInt8(rosimd.BroadcastInt8(10), rosimd.BroadcastInt8(60)),
    ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 10
// 50
// 60
```

Passing `lower` greater than `upper` is a programmer error. Unlike core `ro.Clamp`, which rejects it at construction with a panic, this cannot detect it: the bounds are opaque vectors that a generic operator may not inspect. The composition is `Max(lower)` then `Min(upper)`, so inverted bounds collapse every lane to `upper`.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
