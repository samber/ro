---
name: ReduceMin
slug: reducemin
sourceRef: plugins/exp/simd/reduce.go#L263
type: plugin
category: simd
signatures:
  - "func ReduceMinInt8[V Int8Vector[V]]()"
  - "func ReduceMinInt16[V Int16Vector[V]]()"
  - "func ReduceMinInt32[V Int32Vector[V]]()"
  - "func ReduceMinInt64[V Int64Vector[V]]()"
  - "func ReduceMinUint8[V Uint8Vector[V]]()"
  - "func ReduceMinUint16[V Uint16Vector[V]]()"
  - "func ReduceMinUint32[V Uint32Vector[V]]()"
  - "func ReduceMinUint64[V Uint64Vector[V]]()"
  - "func ReduceMinFloat32[V Float32Vector[V]]()"
  - "func ReduceMinFloat64[V Float64Vector[V]]()"
playUrl:
variantHelpers:
  - plugin#simd#reduceminint8
  - plugin#simd#reduceminint16
  - plugin#simd#reduceminint32
  - plugin#simd#reduceminint64
  - plugin#simd#reduceminuint8
  - plugin#simd#reduceminuint16
  - plugin#simd#reduceminuint32
  - plugin#simd#reduceminuint64
  - plugin#simd#reduceminfloat32
  - plugin#simd#reduceminfloat64
similarHelpers:
  - plugin#simd#reducemax
  - plugin#simd#reducesum
  - plugin#simd#min
position: 190
---

Emits the smallest valid lane of the whole stream on completion.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
    ro.Just[int8](5, 2, 8),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.ReduceMinInt8[rosimd.PartialInt8s](),
)

sub := obs.Subscribe(ro.OnNext(func(smallest int8) {
    fmt.Println(smallest)
}))
defer sub.Unsubscribe()

// 2
```

An empty stream emits nothing, matching `ro.Min`.

For the float types this treats NaN the opposite way from the element-wise `Min`. The comparison is a plain `<`, false for any NaN, so a NaN lane never displaces the accumulator — which is what `ro.Min` does, and agreeing with it is the point. A stream whose first lane is NaN still reduces to NaN, again matching `ro.Min`, because nothing compares less than it.
