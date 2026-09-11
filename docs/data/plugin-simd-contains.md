---
name: Contains
slug: contains
sourceRef: plugins/exp/simd/int8.go#L202
type: plugin
category: simd
signatures:
  - "func (p PartialInt8s) Contains(target PartialInt8s) simd.Mask8s"
  - "func (p PartialInt16s) Contains(target PartialInt16s) simd.Mask16s"
  - "func (p PartialInt32s) Contains(target PartialInt32s) simd.Mask32s"
  - "func (p PartialInt64s) Contains(target PartialInt64s) simd.Mask64s"
  - "func (p PartialUint8s) Contains(target PartialUint8s) simd.Mask8s"
  - "func (p PartialUint16s) Contains(target PartialUint16s) simd.Mask16s"
  - "func (p PartialUint32s) Contains(target PartialUint32s) simd.Mask32s"
  - "func (p PartialUint64s) Contains(target PartialUint64s) simd.Mask64s"
  - "func (p PartialFloat32s) Contains(target PartialFloat32s) simd.Mask32s"
  - "func (p PartialFloat64s) Contains(target PartialFloat64s) simd.Mask64s"
playUrl:
variantHelpers:
  - plugin#simd#containsint8
  - plugin#simd#containsint16
  - plugin#simd#containsint32
  - plugin#simd#containsint64
  - plugin#simd#containsuint8
  - plugin#simd#containsuint16
  - plugin#simd#containsuint32
  - plugin#simd#containsuint64
  - plugin#simd#containsfloat32
  - plugin#simd#containsfloat64
similarHelpers:
  - plugin#simd#select
  - plugin#simd#reducecontains
  - plugin#simd#partial
position: 20
---

Reports, lane by lane, which valid lanes equal `target`.

The result is a **mask** — SIMD's vector of booleans — not a single answer. It is already intersected with the validity mask, so padded lanes are never reported as matches; that intersection is what makes searching for zero correct, since padding is zero-filled.

`Select` consumes the mask:

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](7, 1, 7, 2),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    ro.Map(func(v rosimd.PartialInt8s) []int8 {
        matched := v.Contains(rosimd.BroadcastInt8(7))

        return v.Select(matched, rosimd.BroadcastInt8(0)).Values()
    }),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 7
// 0
// 7
// 0
```

The comparison is lane-for-lane, not lane-against-every-lane. It answers "is this value present" only because `BroadcastInt8` puts one value in every lane of the target; a non-uniform target compares position by position instead.

For the float types, searching for NaN never matches, since NaN compares unequal to everything including itself — which is what Go's own `==` does.

To collapse a whole stream to a single bool instead, use `ReduceContains`.
