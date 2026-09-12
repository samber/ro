---
name: AddWith
slug: addwith
sourceRef: plugins/exp/simd/arithmetic.go#L319
type: plugin
category: simd
signatures:
  - "func AddInt8With[V Int8Vector[V]](other Observable[V])"
  - "func AddInt16With[V Int16Vector[V]](other Observable[V])"
  - "func AddInt32With[V Int32Vector[V]](other Observable[V])"
  - "func AddInt64With[V Int64Vector[V]](other Observable[V])"
  - "func AddUint8With[V Uint8Vector[V]](other Observable[V])"
  - "func AddUint16With[V Uint16Vector[V]](other Observable[V])"
  - "func AddUint32With[V Uint32Vector[V]](other Observable[V])"
  - "func AddUint64With[V Uint64Vector[V]](other Observable[V])"
  - "func AddFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func AddFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#addint8with
  - plugin#simd#addint16with
  - plugin#simd#addint32with
  - plugin#simd#addint64with
  - plugin#simd#adduint8with
  - plugin#simd#adduint16with
  - plugin#simd#adduint32with
  - plugin#simd#adduint64with
  - plugin#simd#addfloat32with
  - plugin#simd#addfloat64with
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

left := ro.Pipe1[int8, rosimd.PartialInt8s](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
)
right := ro.Pipe1[int8, rosimd.PartialInt8s](
    ro.Just[int8](10, 20, 30),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
)

obs := ro.Pipe3[rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    left,
    rosimd.AddInt8With(right),
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
