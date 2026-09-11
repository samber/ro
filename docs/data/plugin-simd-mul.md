---
name: Mul
slug: mul
sourceRef: plugins/exp/simd/arithmetic.go#L240
type: plugin
category: simd
signatures:
  - "func MulInt8[V Int8Vector[V]](operand V)"
  - "func MulInt16[V Int16Vector[V]](operand V)"
  - "func MulInt32[V Int32Vector[V]](operand V)"
  - "func MulUint8[V Uint8Vector[V]](operand V)"
  - "func MulUint16[V Uint16Vector[V]](operand V)"
  - "func MulUint32[V Uint32Vector[V]](operand V)"
  - "func MulFloat32[V Float32Vector[V]](operand V)"
  - "func MulFloat64[V Float64Vector[V]](operand V)"
playUrl:
variantHelpers:
  - plugin#simd#mulint8
  - plugin#simd#mulint16
  - plugin#simd#mulint32
  - plugin#simd#muluint8
  - plugin#simd#muluint16
  - plugin#simd#muluint32
  - plugin#simd#mulfloat32
  - plugin#simd#mulfloat64
similarHelpers:
  - plugin#simd#mulwith
  - plugin#simd#div
  - plugin#simd#add
position: 70
---

Multiplies every lane of every vector in the stream by `operand`.

It exists for every element type except `Int64` and `Uint64`: the standard library provides no 64-bit lane multiply.

The operand is a vector, not a scalar. SIMD has no scalar-operand arithmetic, so widen the value at the call site with the matching `Broadcast` — which is also what lets the type argument be inferred, keeping the call free of an explicit `[rosimd.PartialInt8s]`.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8,
    rosimd.MulInt8(rosimd.BroadcastInt8(3)),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 3
// 6
// 9
```

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
