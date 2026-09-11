---
name: MaxWith
slug: maxwith
sourceRef: plugins/exp/simd/bounds.go#L416
type: plugin
category: simd
signatures:
  - "func MaxInt8With[V Int8Vector[V]](other Observable[V])"
  - "func MaxInt16With[V Int16Vector[V]](other Observable[V])"
  - "func MaxInt32With[V Int32Vector[V]](other Observable[V])"
  - "func MaxInt64With[V Int64Vector[V]](other Observable[V])"
  - "func MaxUint8With[V Uint8Vector[V]](other Observable[V])"
  - "func MaxUint16With[V Uint16Vector[V]](other Observable[V])"
  - "func MaxUint32With[V Uint32Vector[V]](other Observable[V])"
  - "func MaxUint64With[V Uint64Vector[V]](other Observable[V])"
  - "func MaxFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func MaxFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#maxint8with
  - plugin#simd#maxint16with
  - plugin#simd#maxint32with
  - plugin#simd#maxint64with
  - plugin#simd#maxuint8with
  - plugin#simd#maxuint16with
  - plugin#simd#maxuint32with
  - plugin#simd#maxuint64with
  - plugin#simd#maxfloat32with
  - plugin#simd#maxfloat64with
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

left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 50, 100))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 60, 10))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.MaxInt8With(right)(left),
    rosimd.ToScalar[rosimd.PartialInt8s](),
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
