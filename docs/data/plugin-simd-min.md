---
name: Min
slug: min
sourceRef: plugins/exp/simd/bounds.go#L30
type: plugin
category: simd
signatures:
  - "func MinInt8[V Int8Vector[V]](ceiling V)"
  - "func MinInt16[V Int16Vector[V]](ceiling V)"
  - "func MinInt32[V Int32Vector[V]](ceiling V)"
  - "func MinInt64[V Int64Vector[V]](ceiling V)"
  - "func MinUint8[V Uint8Vector[V]](ceiling V)"
  - "func MinUint16[V Uint16Vector[V]](ceiling V)"
  - "func MinUint32[V Uint32Vector[V]](ceiling V)"
  - "func MinUint64[V Uint64Vector[V]](ceiling V)"
  - "func MinFloat32[V Float32Vector[V]](ceiling V)"
  - "func MinFloat64[V Float64Vector[V]](ceiling V)"
playUrl:
variantHelpers:
  - plugin#simd#minint8
  - plugin#simd#minint16
  - plugin#simd#minint32
  - plugin#simd#minint64
  - plugin#simd#minuint8
  - plugin#simd#minuint16
  - plugin#simd#minuint32
  - plugin#simd#minuint64
  - plugin#simd#minfloat32
  - plugin#simd#minfloat64
similarHelpers:
  - plugin#simd#minwith
  - plugin#simd#max
  - plugin#simd#clamp
  - plugin#simd#reducemin
position: 90
---

Keeps the smaller of each lane and the matching lane of `ceiling`.

This is element-wise, one vector out per vector in — unlike `ro.Min`, which aggregates a whole stream into a single value. `ReduceMin` is the aggregating counterpart.

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
    rosimd.MinInt8(rosimd.BroadcastInt8(60)),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 1
// 50
// 60
```

For the float types, a lane where either operand is NaN yields NaN, matching Go's own `min` builtin. Hardware disagrees about this — x86 discards NaN, arm64 propagates it — so the `Partial` types detect NaN lanes and force the result, which makes every architecture agree. A standard library `simd.Float64s` passed through the same operator uses the raw instruction and does not carry that guarantee.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
