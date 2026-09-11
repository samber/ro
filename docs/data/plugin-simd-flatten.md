---
name: Flatten
slug: flatten
sourceRef: plugins/exp/simd/vectorize.go#L316
type: plugin
category: simd
signatures:
  - "func FlattenInt8[V LaneStore[int8]](source Observable[V]) Observable[int8]"
  - "func FlattenInt16[V LaneStore[int16]](source Observable[V]) Observable[int16]"
  - "func FlattenInt32[V LaneStore[int32]](source Observable[V]) Observable[int32]"
  - "func FlattenInt64[V LaneStore[int64]](source Observable[V]) Observable[int64]"
  - "func FlattenUint8[V LaneStore[uint8]](source Observable[V]) Observable[uint8]"
  - "func FlattenUint16[V LaneStore[uint16]](source Observable[V]) Observable[uint16]"
  - "func FlattenUint32[V LaneStore[uint32]](source Observable[V]) Observable[uint32]"
  - "func FlattenUint64[V LaneStore[uint64]](source Observable[V]) Observable[uint64]"
  - "func FlattenFloat32[V LaneStore[float32]](source Observable[V]) Observable[float32]"
  - "func FlattenFloat64[V LaneStore[float64]](source Observable[V]) Observable[float64]"
playUrl:
variantHelpers:
  - plugin#simd#flattenint8
  - plugin#simd#flattenint16
  - plugin#simd#flattenint32
  - plugin#simd#flattenint64
  - plugin#simd#flattenuint8
  - plugin#simd#flattenuint16
  - plugin#simd#flattenuint32
  - plugin#simd#flattenuint64
  - plugin#simd#flattenfloat32
  - plugin#simd#flattenfloat64
similarHelpers:
  - plugin#simd#toscalar
  - plugin#simd#vectorize
  - plugin#simd#reducesum
position: 47
---

Hands each vector's valid lanes back one at a time, turning a vector stream into a scalar stream.

It is `ToScalar` followed by `ro.Flatten` in a single stage: where `ToScalar` emits one slice per vector, this emits one value per lane. Padded lanes are never emitted, so a stream that goes through `Vectorize` and back out through this arrives unchanged.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
    ro.Just[int8](1, 2, 3, 4, 5),
    rosimd.VectorizeInt8,
    rosimd.FlattenInt8,
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

Every lane of one vector carries that vector's own context onward, so context propagation survives the round trip.

Like `ToScalar`, its constraint asks only that a vector can report its lanes, so it accepts the standard library's vector types too.

It is not a curried operator, so the type argument is inferred from the surrounding `Pipe`.
