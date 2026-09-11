---
name: SubWith
slug: subwith
sourceRef: plugins/exp/simd/arithmetic.go#L416
type: plugin
category: simd
signatures:
  - "func SubInt8With[V Int8Vector[V]](other Observable[V])"
  - "func SubInt16With[V Int16Vector[V]](other Observable[V])"
  - "func SubInt32With[V Int32Vector[V]](other Observable[V])"
  - "func SubInt64With[V Int64Vector[V]](other Observable[V])"
  - "func SubUint8With[V Uint8Vector[V]](other Observable[V])"
  - "func SubUint16With[V Uint16Vector[V]](other Observable[V])"
  - "func SubUint32With[V Uint32Vector[V]](other Observable[V])"
  - "func SubUint64With[V Uint64Vector[V]](other Observable[V])"
  - "func SubFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func SubFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#subint8with
  - plugin#simd#subint16with
  - plugin#simd#subint32with
  - plugin#simd#subint64with
  - plugin#simd#subuint8with
  - plugin#simd#subuint16with
  - plugin#simd#subuint32with
  - plugin#simd#subuint64with
  - plugin#simd#subfloat32with
  - plugin#simd#subfloat64with
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

left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 20, 30))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 2, 3))

obs := ro.Pipe3[rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    left,
    rosimd.SubInt8With(right),
    rosimd.ToScalar[rosimd.PartialInt8s](),
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
