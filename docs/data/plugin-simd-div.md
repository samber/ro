---
name: Div
slug: div
sourceRef: plugins/exp/simd/arithmetic.go#L309
type: plugin
category: simd
signatures:
  - "func DivFloat32[V Float32Vector[V]](operand V)"
  - "func DivFloat64[V Float64Vector[V]](operand V)"
playUrl:
variantHelpers:
  - plugin#simd#divfloat32
  - plugin#simd#divfloat64
similarHelpers:
  - plugin#simd#divwith
  - plugin#simd#mul
position: 80
---

Divides every lane of every vector in the stream by `operand`.

It exists for `Float32` and `Float64` only: the standard library provides no lane-wise integer division.

The operand is a vector, not a scalar. SIMD has no scalar-operand arithmetic, so widen the value at the call site with the matching `Broadcast` — which is also what lets the type argument be inferred, keeping the call free of an explicit `[rosimd.PartialInt8s]`.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe4[float64, rosimd.PartialFloat64s, rosimd.PartialFloat64s, []float64, float64](
    ro.Just[float64](1, 2, 3),
    rosimd.VectorizeFloat64,
    rosimd.DivFloat64(rosimd.BroadcastFloat64(2)),
    ro.Map(func(v rosimd.PartialFloat64s) []float64 { return v.Values() }),
    ro.Flatten[float64](),
)

sub := obs.Subscribe(ro.OnNext(func(value float64) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 0.5
// 1
// 1.5
```

Dividing by zero yields ±Inf and 0/0 yields NaN, exactly as Go's own float division does. Neither is an error, and neither is suppressed.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
