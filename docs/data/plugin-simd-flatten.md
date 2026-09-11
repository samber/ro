---
name: Flatten
slug: flatten
sourceRef: plugins/exp/simd/vectorize.go#L274
type: plugin
category: simd
signatures:
  - "func Flatten[T any, V LaneStore[T]](source Observable[V]) Observable[T]"
playUrl:
variantHelpers:
  - plugin#simd#flatten
similarHelpers:
  - plugin#simd#toscalar
  - plugin#simd#vectorize
  - plugin#simd#reducesum
position: 47
---

Hands each vector's valid lanes back one at a time, turning a vector stream into a scalar stream.

It is `ToScalar` followed by `ro.Flatten` in a single stage: where `ToScalar` emits one slice per vector, this emits one value per lane. Padded lanes are never emitted, so a stream that goes through `Vectorize` and back out through this arrives unchanged.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int8](
    ro.Just[int8](1, 2, 3, 4, 5),
    rosimd.VectorizeInt8,
    rosimd.Flatten,
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 1
// 2
// 3
// 4
// 5
```

Every lane of one vector carries that vector's own context onward, so context propagation survives the round trip.

Like `ToScalar`, its constraint asks only that a vector can report its lanes, so it accepts the standard library's vector types too.

It is not a curried operator, so the type argument is inferred from the surrounding `Pipe`.
