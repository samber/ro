---
name: Max
slug: max
sourceRef: plugins/exp/simd/int8.go#L308
type: plugin
category: simd
signatures:
  - "func MaxInt8[V Int8Vector[V]](floor V)"
  - "func MaxInt16[V Int16Vector[V]](floor V)"
  - "func MaxInt32[V Int32Vector[V]](floor V)"
  - "func MaxInt64[V Int64Vector[V]](floor V)"
  - "func MaxUint8[V Uint8Vector[V]](floor V)"
  - "func MaxUint16[V Uint16Vector[V]](floor V)"
  - "func MaxUint32[V Uint32Vector[V]](floor V)"
  - "func MaxUint64[V Uint64Vector[V]](floor V)"
  - "func MaxFloat32[V Float32Vector[V]](floor V)"
  - "func MaxFloat64[V Float64Vector[V]](floor V)"
playUrl:
variantHelpers:
  - plugin#simd#maxint8
  - plugin#simd#maxint16
  - plugin#simd#maxint32
  - plugin#simd#maxint64
  - plugin#simd#maxuint8
  - plugin#simd#maxuint16
  - plugin#simd#maxuint32
  - plugin#simd#maxuint64
  - plugin#simd#maxfloat32
  - plugin#simd#maxfloat64
similarHelpers:
  - plugin#simd#maxwith
  - plugin#simd#min
  - plugin#simd#clamp
  - plugin#simd#reducemax
position: 100
---

Keeps the larger of each lane and the matching lane of `floor`.

This is element-wise. `ReduceMax` is the aggregating counterpart.

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
    rosimd.MaxInt8(rosimd.BroadcastInt8(40)),
    ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 40
// 50
// 100
```

For the float types, NaN propagates as it does in `Min`.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
