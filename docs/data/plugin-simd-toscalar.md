---
name: ToScalar
slug: toscalar
sourceRef: plugins/exp/simd/vectorize.go#L252
type: plugin
category: simd
signatures:
  - "func ToScalar[V LaneStore[T], T any]()"
playUrl:
variantHelpers:
  - plugin#simd#toscalar
similarHelpers:
  - plugin#simd#flatten
  - plugin#simd#vectorize
  - plugin#simd#partial
position: 45
---

Hands each vector's valid lanes back as a slice, one slice per vector.

It is the exit from vector space, the counterpart of `Vectorize`. A short final batch yields a correspondingly short slice — padded lanes are never included — so the slices concatenated are exactly the stream that went in.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, []int8](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.ToScalar[rosimd.PartialInt8s](),
)

sub := obs.Subscribe(ro.OnNext(func(lanes []int8) {
    fmt.Println(lanes)
}))
defer sub.Unsubscribe()

// [1 2 3]
```

Pair it with `ro.Flatten` to get a scalar stream back, or use `Flatten` to do both in one stage.

Its constraint asks only that a vector can report its lanes, not that it can do arithmetic, so it accepts the standard library's vector types as well — `simd.Int64s` and `simd.Uint64s` included, which the arithmetic operators reject for want of `Min` and `Max`.

It is not a curried operator, so the type argument is inferred from the surrounding `Pipe`.
