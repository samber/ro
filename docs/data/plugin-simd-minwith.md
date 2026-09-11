---
name: MinWith
slug: minwith
sourceRef: plugins/exp/simd/bounds.go#L346
type: plugin
category: simd
signatures:
  - "func MinWithInt8[V Int8Vector[V]](other Observable[V])"
  - "func MinWithInt16[V Int16Vector[V]](other Observable[V])"
  - "func MinWithInt32[V Int32Vector[V]](other Observable[V])"
  - "func MinWithInt64[V Int64Vector[V]](other Observable[V])"
  - "func MinWithUint8[V Uint8Vector[V]](other Observable[V])"
  - "func MinWithUint16[V Uint16Vector[V]](other Observable[V])"
  - "func MinWithUint32[V Uint32Vector[V]](other Observable[V])"
  - "func MinWithUint64[V Uint64Vector[V]](other Observable[V])"
  - "func MinWithFloat32[V Float32Vector[V]](other Observable[V])"
  - "func MinWithFloat64[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#minwithint8
  - plugin#simd#minwithint16
  - plugin#simd#minwithint32
  - plugin#simd#minwithint64
  - plugin#simd#minwithuint8
  - plugin#simd#minwithuint16
  - plugin#simd#minwithuint32
  - plugin#simd#minwithuint64
  - plugin#simd#minwithfloat32
  - plugin#simd#minwithfloat64
similarHelpers:
  - plugin#simd#min
  - plugin#simd#maxwith
position: 160
---

Keeps the smaller of each lane pair from two vector streams.

It is the curried, two-stream counterpart of `Min`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

left := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](1, 50, 100))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](10, 10, 10))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.MinWithInt8(right)(left),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 1
// 10
// 10
```

For the float types, NaN propagates exactly as it does in `Min`.
