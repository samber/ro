---
name: DivWith
slug: divwith
sourceRef: plugins/exp/simd/arithmetic.go#L549
type: plugin
category: simd
signatures:
  - "func DivFloat32With[V Float32Vector[V]](other Observable[V])"
  - "func DivFloat64With[V Float64Vector[V]](other Observable[V])"
playUrl:
variantHelpers:
  - plugin#simd#divfloat32with
  - plugin#simd#divfloat64with
similarHelpers:
  - plugin#simd#div
  - plugin#simd#mulwith
position: 150
---

Divides this vector stream by another, lane by lane.

It exists for `Float32` and `Float64` only.

It is the curried, two-stream counterpart of `Div`, following the same naming convention as `ro.ZipWith` and `ro.MergeWith`. Vectors are paired in order, and the stream ends as soon as either side completes and its buffer is drained.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

left := rosimd.VectorizeFloat64[rosimd.PartialFloat64s]()(ro.Just[float64](10, 20, 30))
right := rosimd.VectorizeFloat64[rosimd.PartialFloat64s]()(ro.Just[float64](2, 4, 5))

obs := ro.Pipe3[rosimd.PartialFloat64s, rosimd.PartialFloat64s, []float64, float64](
    left,
    rosimd.DivFloat64With(right),
    rosimd.ToScalar[rosimd.PartialFloat64s](),
    ro.Flatten[float64](),
)

sub := obs.Subscribe(ro.OnNext(func(value float64) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 5
// 5
// 6
```
