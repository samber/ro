---
name: SubWith
slug: subwith
sourceRef: plugins/exp/simd/arithmetic.go#L416
type: plugin
category: simd
signatures:
  - "func SubWithInt8[V Int8Vector[V]](other Observable[V])"
  - "func SubWithInt16[V Int16Vector[V]](other Observable[V])"
  - "func SubWithInt32[V Int32Vector[V]](other Observable[V])"
  - "func SubWithInt64[V Int64Vector[V]](other Observable[V])"
  - "func SubWithUint8[V Uint8Vector[V]](other Observable[V])"
  - "func SubWithUint16[V Uint16Vector[V]](other Observable[V])"
  - "func SubWithUint32[V Uint32Vector[V]](other Observable[V])"
  - "func SubWithUint64[V Uint64Vector[V]](other Observable[V])"
  - "func SubWithFloat32[V Float32Vector[V]](other Observable[V])"
  - "func SubWithFloat64[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#subwithint8
  - plugin#simd#subwithint16
  - plugin#simd#subwithint32
  - plugin#simd#subwithint64
  - plugin#simd#subwithuint8
  - plugin#simd#subwithuint16
  - plugin#simd#subwithuint32
  - plugin#simd#subwithuint64
  - plugin#simd#subwithfloat32
  - plugin#simd#subwithfloat64
similarHelpers:
  - plugin#simd#sub
  - plugin#simd#addwith
position: 130
---

Subtracts another vector stream from this one, lane by lane.

It is the curried, two-stream counterpart of `Sub`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

left := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](10, 20, 30))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s](ro.Just[int8](1, 2, 3))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.SubWithInt8(right)(left),
    rosimd.ToScalarInt8,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 9
// 18
// 27
```
