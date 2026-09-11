---
name: Vectorize
slug: vectorize
sourceRef: plugins/exp/simd/vectorize.go#L39
type: plugin
category: simd
signatures:
  - "func VectorizeInt8[V Int8Buffer[V]](source Observable[int8]) Observable[V]"
  - "func VectorizeInt16[V Int16Buffer[V]](source Observable[int16]) Observable[V]"
  - "func VectorizeInt32[V Int32Buffer[V]](source Observable[int32]) Observable[V]"
  - "func VectorizeInt64[V Int64Buffer[V]](source Observable[int64]) Observable[V]"
  - "func VectorizeUint8[V Uint8Buffer[V]](source Observable[uint8]) Observable[V]"
  - "func VectorizeUint16[V Uint16Buffer[V]](source Observable[uint16]) Observable[V]"
  - "func VectorizeUint32[V Uint32Buffer[V]](source Observable[uint32]) Observable[V]"
  - "func VectorizeUint64[V Uint64Buffer[V]](source Observable[uint64]) Observable[V]"
  - "func VectorizeFloat32[V Float32Buffer[V]](source Observable[float32]) Observable[V]"
  - "func VectorizeFloat64[V Float64Buffer[V]](source Observable[float64]) Observable[V]"
playUrl:
variantHelpers:
  - plugin#simd#vectorizeint8
  - plugin#simd#vectorizeint16
  - plugin#simd#vectorizeint32
  - plugin#simd#vectorizeint64
  - plugin#simd#vectorizeuint8
  - plugin#simd#vectorizeuint16
  - plugin#simd#vectorizeuint32
  - plugin#simd#vectorizeuint64
  - plugin#simd#vectorizefloat32
  - plugin#simd#vectorizefloat64
similarHelpers:
  - plugin#simd#toscalar
  - plugin#simd#flatten
  - plugin#simd#partial
  - plugin#simd#reducesum
  - plugin#simd#broadcast
position: 40
---

Batches a scalar stream into vectors, so the operators downstream can work on a whole register at a time.

A full vector is emitted every time the buffer fills, and on completion one final `Partial` vector holds whatever is left. The lane width is discovered from the running architecture on subscription, not hard-coded.

Leaving vector space again is `ToScalar`, which hands each vector's lanes back as a slice, or `Flatten`, which emits them one at a time. The `Reduce` operators are the other way out:

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](1, 2, 3, 4, 5),
    rosimd.VectorizeInt8,
    rosimd.ToScalar,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 1
// 2
// 3
// 4
// 5
```

It is not a curried operator: taking the source directly lets the type argument be inferred from the surrounding `Pipe`.

It produces `Partial` types only. The standard library's vector types expose no constructor method, and a generic function cannot reach the `simd.LoadXxx` package functions.
