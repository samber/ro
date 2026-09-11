---
name: ToScalar
slug: toscalar
sourceRef: plugins/exp/simd/vectorize.go#L250
type: plugin
category: simd
signatures:
  - "func ToScalarInt8[V LaneStore[int8]](source Observable[V]) Observable[[]int8]"
  - "func ToScalarInt16[V LaneStore[int16]](source Observable[V]) Observable[[]int16]"
  - "func ToScalarInt32[V LaneStore[int32]](source Observable[V]) Observable[[]int32]"
  - "func ToScalarInt64[V LaneStore[int64]](source Observable[V]) Observable[[]int64]"
  - "func ToScalarUint8[V LaneStore[uint8]](source Observable[V]) Observable[[]uint8]"
  - "func ToScalarUint16[V LaneStore[uint16]](source Observable[V]) Observable[[]uint16]"
  - "func ToScalarUint32[V LaneStore[uint32]](source Observable[V]) Observable[[]uint32]"
  - "func ToScalarUint64[V LaneStore[uint64]](source Observable[V]) Observable[[]uint64]"
  - "func ToScalarFloat32[V LaneStore[float32]](source Observable[V]) Observable[[]float32]"
  - "func ToScalarFloat64[V LaneStore[float64]](source Observable[V]) Observable[[]float64]"
playUrl:
variantHelpers:
  - plugin#simd#toscalarint8
  - plugin#simd#toscalarint16
  - plugin#simd#toscalarint32
  - plugin#simd#toscalarint64
  - plugin#simd#toscalaruint8
  - plugin#simd#toscalaruint16
  - plugin#simd#toscalaruint32
  - plugin#simd#toscalaruint64
  - plugin#simd#toscalarfloat32
  - plugin#simd#toscalarfloat64
similarHelpers:
  - plugin#simd#flatten
  - plugin#simd#vectorize
  - plugin#simd#partial
position: 45
---

Hands each vector's valid lanes back as a slice, one slice per vector.

It is the exit from vector space, the counterpart of `Vectorize`. A short final batch yields a correspondingly short slice — padded lanes are never included — so the slices concatenated are exactly the stream that went in.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, []int8](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8,
    rosimd.ToScalarInt8,
)

sub := obs.Subscribe(ro.OnNext(func(lanes []int8) {
    fmt.Println(lanes)
}))
defer sub.Unsubscribe()

// [1 2 3]
```

Pair it with `ro.Flatten` to get a scalar stream back, or use `Flatten` to do both in one stage.

Its constraint asks only that a vector can report its lanes, not that it can do arithmetic, so it accepts the standard library's vector types as well — `simd.Int64s` and `simd.Uint64s` included, which the arithmetic operators reject for want of `Min` and `Max`.

It is not a curried operator, so the type argument is inferred from the surrounding `Pipe`.
