---
name: Add
slug: add
sourceRef: plugins/exp/simd/arithmetic.go#L33
type: plugin
category: simd
signatures:
  - "func AddInt8[V Int8Vector[V]](operand V)"
  - "func AddInt16[V Int16Vector[V]](operand V)"
  - "func AddInt32[V Int32Vector[V]](operand V)"
  - "func AddInt64[V Int64Vector[V]](operand V)"
  - "func AddUint8[V Uint8Vector[V]](operand V)"
  - "func AddUint16[V Uint16Vector[V]](operand V)"
  - "func AddUint32[V Uint32Vector[V]](operand V)"
  - "func AddUint64[V Uint64Vector[V]](operand V)"
  - "func AddFloat32[V Float32Vector[V]](operand V)"
  - "func AddFloat64[V Float64Vector[V]](operand V)"
playUrl:
variantHelpers:
  - plugin#simd#addint8
  - plugin#simd#addint16
  - plugin#simd#addint32
  - plugin#simd#addint64
  - plugin#simd#adduint8
  - plugin#simd#adduint16
  - plugin#simd#adduint32
  - plugin#simd#adduint64
  - plugin#simd#addfloat32
  - plugin#simd#addfloat64
similarHelpers:
  - plugin#simd#addwith
  - plugin#simd#sub
  - plugin#simd#mul
  - plugin#simd#broadcast
position: 50
signature: "func Add(d time.Duration) func(destination ro.Observable[time.Time]) ro.Observable[time.Time] {"
---

Adds `operand` to every lane of every vector in the stream.

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
    rosimd.AddInt8(rosimd.BroadcastInt8(10)),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 11
// 12
// 13
```

Overflow wraps in the element type, as Go's own `+` does.

The operator is generic over an interface satisfied by both this package's `Partial` types and the standard library's own vector types, so a stream of `simd.Int8s` flows through it just as well — broadcast the operand with `simd.BroadcastInt8s` in that case.
