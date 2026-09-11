---
name: MaxWith
slug: maxwith
sourceRef: plugins/exp/simd/bounds.go#L416
type: plugin
category: simd
signatures:
  - "func MaxWithInt8[V Int8Vector[V]](other Observable[V])"
  - "func MaxWithInt16[V Int16Vector[V]](other Observable[V])"
  - "func MaxWithInt32[V Int32Vector[V]](other Observable[V])"
  - "func MaxWithInt64[V Int64Vector[V]](other Observable[V])"
  - "func MaxWithUint8[V Uint8Vector[V]](other Observable[V])"
  - "func MaxWithUint16[V Uint16Vector[V]](other Observable[V])"
  - "func MaxWithUint32[V Uint32Vector[V]](other Observable[V])"
  - "func MaxWithUint64[V Uint64Vector[V]](other Observable[V])"
  - "func MaxWithFloat32[V Float32Vector[V]](other Observable[V])"
  - "func MaxWithFloat64[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#maxwithint8
  - plugin#simd#maxwithint16
  - plugin#simd#maxwithint32
  - plugin#simd#maxwithint64
  - plugin#simd#maxwithuint8
  - plugin#simd#maxwithuint16
  - plugin#simd#maxwithuint32
  - plugin#simd#maxwithuint64
  - plugin#simd#maxwithfloat32
  - plugin#simd#maxwithfloat64
similarHelpers:
  - plugin#simd#max
  - plugin#simd#minwith
position: 170
---

Keeps the larger of each lane pair from two vector streams.

It is the curried, two-stream counterpart of `Max`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

left := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](1, 50, 100))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](10, 60, 10))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.MaxWithInt8(right)(left),
    ro.Map(func(v rosimd.PartialInt8s) []int8 { return v.Values() }),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 10
// 60
// 100
```

For the float types, NaN propagates exactly as it does in `Max`.
