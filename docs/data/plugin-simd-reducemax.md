---
name: ReduceMax
slug: reducemax
sourceRef: plugins/exp/simd/reduce.go#L367
type: plugin
category: simd
signatures:
  - "func ReduceMaxInt8[V Int8Vector[V]](source Observable[V]) Observable[int8]"
  - "func ReduceMaxInt16[V Int16Vector[V]](source Observable[V]) Observable[int16]"
  - "func ReduceMaxInt32[V Int32Vector[V]](source Observable[V]) Observable[int32]"
  - "func ReduceMaxInt64[V Int64Vector[V]](source Observable[V]) Observable[int64]"
  - "func ReduceMaxUint8[V Uint8Vector[V]](source Observable[V]) Observable[uint8]"
  - "func ReduceMaxUint16[V Uint16Vector[V]](source Observable[V]) Observable[uint16]"
  - "func ReduceMaxUint32[V Uint32Vector[V]](source Observable[V]) Observable[uint32]"
  - "func ReduceMaxUint64[V Uint64Vector[V]](source Observable[V]) Observable[uint64]"
  - "func ReduceMaxFloat32[V Float32Vector[V]](source Observable[V]) Observable[float32]"
  - "func ReduceMaxFloat64[V Float64Vector[V]](source Observable[V]) Observable[float64]"
playUrl:
variantHelpers:
  - plugin#simd#reducemaxint8
  - plugin#simd#reducemaxint16
  - plugin#simd#reducemaxint32
  - plugin#simd#reducemaxint64
  - plugin#simd#reducemaxuint8
  - plugin#simd#reducemaxuint16
  - plugin#simd#reducemaxuint32
  - plugin#simd#reducemaxuint64
  - plugin#simd#reducemaxfloat32
  - plugin#simd#reducemaxfloat64
similarHelpers:
  - plugin#simd#reducemin
  - plugin#simd#reducesum
  - plugin#simd#max
position: 200
---

Emits the largest valid lane of the whole stream on completion.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
    ro.Just[int8](5, 2, 8),
    rosimd.VectorizeInt8,
    rosimd.ReduceMaxInt8,
)

sub := obs.Subscribe(ro.OnNext(func(largest int8) {
    fmt.Println(largest)
}))
defer sub.Unsubscribe()

// 8
```

An empty stream emits nothing. For the float types, NaN never displaces the accumulator — see `ReduceMin` for why this differs from the element-wise `Max`.
