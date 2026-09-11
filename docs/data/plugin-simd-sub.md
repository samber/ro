---
name: Sub
slug: sub
sourceRef: plugins/exp/simd/arithmetic.go#L164
type: plugin
category: simd
signatures:
  - "func SubInt8[V Int8Vector[V]](operand V)"
  - "func SubInt16[V Int16Vector[V]](operand V)"
  - "func SubInt32[V Int32Vector[V]](operand V)"
  - "func SubInt64[V Int64Vector[V]](operand V)"
  - "func SubUint8[V Uint8Vector[V]](operand V)"
  - "func SubUint16[V Uint16Vector[V]](operand V)"
  - "func SubUint32[V Uint32Vector[V]](operand V)"
  - "func SubUint64[V Uint64Vector[V]](operand V)"
  - "func SubFloat32[V Float32Vector[V]](operand V)"
  - "func SubFloat64[V Float64Vector[V]](operand V)"
playUrl:
variantHelpers:
  - plugin#simd#subint8
  - plugin#simd#subint16
  - plugin#simd#subint32
  - plugin#simd#subint64
  - plugin#simd#subuint8
  - plugin#simd#subuint16
  - plugin#simd#subuint32
  - plugin#simd#subuint64
  - plugin#simd#subfloat32
  - plugin#simd#subfloat64
similarHelpers:
  - plugin#simd#subwith
  - plugin#simd#add
  - plugin#simd#mul
position: 60
---

Subtracts `operand` from every lane of every vector in the stream.

The operand is a vector, not a scalar. SIMD has no scalar-operand arithmetic, so widen the value at the call site with the matching `Broadcast` — which is also what lets the type argument be inferred, keeping the call free of an explicit `[rosimd.PartialInt8s]`.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](10, 20, 30),
    rosimd.VectorizeInt8,
    rosimd.SubInt8(rosimd.BroadcastInt8(5)),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 5
// 15
// 25
```

Underflow wraps in the element type, as Go's own `-` does. On the unsigned types that means a result below zero becomes a large positive value.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
