---
name: MulWith
slug: mulwith
sourceRef: plugins/exp/simd/arithmetic.go#L486
type: plugin
category: simd
signatures:
  - "func MulWithInt8[V Int8Vector[V]](other Observable[V])"
  - "func MulWithInt16[V Int16Vector[V]](other Observable[V])"
  - "func MulWithInt32[V Int32Vector[V]](other Observable[V])"
  - "func MulWithUint8[V Uint8Vector[V]](other Observable[V])"
  - "func MulWithUint16[V Uint16Vector[V]](other Observable[V])"
  - "func MulWithUint32[V Uint32Vector[V]](other Observable[V])"
  - "func MulWithFloat32[V Float32Vector[V]](other Observable[V])"
  - "func MulWithFloat64[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#mulwithint8
  - plugin#simd#mulwithint16
  - plugin#simd#mulwithint32
  - plugin#simd#mulwithuint8
  - plugin#simd#mulwithuint16
  - plugin#simd#mulwithuint32
  - plugin#simd#mulwithfloat32
  - plugin#simd#mulwithfloat64
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

left := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](1, 2, 3))
right := rosimd.VectorizeInt8[rosimd.PartialInt8s]()(ro.Just[int8](10, 20, 30))

obs := ro.Pipe2[rosimd.PartialInt8s, []int8, int8](
    rosimd.MulWithInt8(right)(left),
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
