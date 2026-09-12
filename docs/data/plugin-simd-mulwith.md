---
name: MulWith
slug: mulwith
sourceRef: plugins/exp/simd/arithmetic.go#L486
type: plugin
category: simd
signatures:
  - "func MulInt8With[V Int8Vector[V]](other Observable[V])"
  - "func MulInt16With[V Int16Vector[V]](other Observable[V])"
  - "func MulInt32With[V Int32Vector[V]](other Observable[V])"
  - "func MulUint8With[V Uint8Vector[V]](other Observable[V])"
  - "func MulUint16With[V Uint16Vector[V]](other Observable[V])"
  - "func MulUint32With[V Uint32Vector[V]](other Observable[V])"
  - "func MulFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func MulFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#mulint8with
  - plugin#simd#mulint16with
  - plugin#simd#mulint32with
  - plugin#simd#muluint8with
  - plugin#simd#muluint16with
  - plugin#simd#muluint32with
  - plugin#simd#mulfloat32with
  - plugin#simd#mulfloat64with
similarHelpers:
  - plugin#simd#mul
  - plugin#simd#divwith
position: 140
---

Multiplies this vector stream by another, lane by lane.

It exists for every element type except `Int64` and `Uint64`, which have no lane multiply.

It is the curried, two-stream counterpart of `Mul`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

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
    rosimd.MulInt8With(right),
    rosimd.ToScalar[rosimd.PartialInt8s](),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 10
// 40
// 90
```
