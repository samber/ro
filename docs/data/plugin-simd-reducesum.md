---
name: ReduceSum
slug: reducesum
sourceRef: plugins/exp/simd/reduce.go#L39
type: plugin
category: simd
signatures:
  - "func ReduceSumInt8[V Int8Vector[V]]()"
  - "func ReduceSumInt16[V Int16Vector[V]]()"
  - "func ReduceSumInt32[V Int32Vector[V]]()"
  - "func ReduceSumInt64[V Int64Vector[V]]()"
  - "func ReduceSumUint8[V Uint8Vector[V]]()"
  - "func ReduceSumUint16[V Uint16Vector[V]]()"
  - "func ReduceSumUint32[V Uint32Vector[V]]()"
  - "func ReduceSumUint64[V Uint64Vector[V]]()"
  - "func ReduceSumFloat32[V Float32Vector[V]]()"
  - "func ReduceSumFloat64[V Float64Vector[V]]()"
playUrl:
variantHelpers:
  - plugin#simd#reducesumint8
  - plugin#simd#reducesumint16
  - plugin#simd#reducesumint32
  - plugin#simd#reducesumint64
  - plugin#simd#reducesumuint8
  - plugin#simd#reducesumuint16
  - plugin#simd#reducesumuint32
  - plugin#simd#reducesumuint64
  - plugin#simd#reducesumfloat32
  - plugin#simd#reducesumfloat64
similarHelpers:
  - plugin#simd#reducemin
  - plugin#simd#reducemax
  - plugin#simd#vectorize
position: 180
---

Totals every valid lane of the stream and emits the sum on completion.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
    ro.Just[int8](1, 2, 3, 4, 5),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.ReduceSumInt8[rosimd.PartialInt8s](),
)

sub := obs.Subscribe(ro.OnNext(func(total int8) {
    fmt.Println(total)
}))
defer sub.Unsubscribe()

// 15
```

The sum accumulates in the element type and wraps on overflow, exactly as `ro.Sum` does — it does not promote to a wider type. An empty stream emits zero.

The vector type is given at the call site, since currying puts it out of inference's reach.
