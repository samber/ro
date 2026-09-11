---
name: MinWith
slug: minwith
sourceRef: plugins/exp/simd/bounds.go#L346
type: plugin
category: simd
signatures:
  - "func MinInt8With[V Int8Vector[V]](other Observable[V])"
  - "func MinInt16With[V Int16Vector[V]](other Observable[V])"
  - "func MinInt32With[V Int32Vector[V]](other Observable[V])"
  - "func MinInt64With[V Int64Vector[V]](other Observable[V])"
  - "func MinUint8With[V Uint8Vector[V]](other Observable[V])"
  - "func MinUint16With[V Uint16Vector[V]](other Observable[V])"
  - "func MinUint32With[V Uint32Vector[V]](other Observable[V])"
  - "func MinUint64With[V Uint64Vector[V]](other Observable[V])"
  - "func MinFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func MinFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#minint8with
  - plugin#simd#minint16with
  - plugin#simd#minint32with
  - plugin#simd#minint64with
  - plugin#simd#minuint8with
  - plugin#simd#minuint16with
  - plugin#simd#minuint32with
  - plugin#simd#minuint64with
  - plugin#simd#minfloat32with
  - plugin#simd#minfloat64with
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

left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 50, 100))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 10, 10))

obs := ro.Pipe3[rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    left,
    rosimd.MinInt8With(right),
    rosimd.ToScalar[rosimd.PartialInt8s](),
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
