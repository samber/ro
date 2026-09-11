---
name: AddWith
slug: addwith
sourceRef: plugins/exp/simd/arithmetic.go#L319
type: plugin
category: simd
signatures:
  - "func AddWithInt8[V Int8Vector[V]](other Observable[V])"
  - "func AddWithInt16[V Int16Vector[V]](other Observable[V])"
  - "func AddWithInt32[V Int32Vector[V]](other Observable[V])"
  - "func AddWithInt64[V Int64Vector[V]](other Observable[V])"
  - "func AddWithUint8[V Uint8Vector[V]](other Observable[V])"
  - "func AddWithUint16[V Uint16Vector[V]](other Observable[V])"
  - "func AddWithUint32[V Uint32Vector[V]](other Observable[V])"
  - "func AddWithUint64[V Uint64Vector[V]](other Observable[V])"
  - "func AddWithFloat32[V Float32Vector[V]](other Observable[V])"
  - "func AddWithFloat64[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#addwithint8
  - plugin#simd#addwithint16
  - plugin#simd#addwithint32
  - plugin#simd#addwithint64
  - plugin#simd#addwithuint8
  - plugin#simd#addwithuint16
  - plugin#simd#addwithuint32
  - plugin#simd#addwithuint64
  - plugin#simd#addwithfloat32
  - plugin#simd#addwithfloat64
similarHelpers:
  - plugin#simd#add
  - plugin#simd#subwith
  - plugin#simd#mulwith
position: 120
---

Adds another vector stream to this one, lane by lane.

It is the curried, two-stream counterpart of `Add`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 2, 3))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 20, 30))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.AddWithInt8(right)(left),
    rosimd.ToScalar[rosimd.PartialInt8s](),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 11
// 22
// 33
```
