---
name: PartialInt8s
slug: partial
sourceRef: plugins/exp/simd/int8.go#L59
type: plugin
category: simd
signatures:
  - "func (p PartialInt8s) Values() []int8"
  - "func (p PartialInt8s) Count() int"
  - "func (p PartialInt8s) Len() int"
  - "func (p PartialInt8s) StorePart(dst []int8) int"
  - "func (p PartialInt8s) Sum() int8"
  - "func (p PartialInt8s) Add(other PartialInt8s) PartialInt8s"
  - "func (p PartialInt8s) Sub(other PartialInt8s) PartialInt8s"
  - "func (p PartialInt8s) Mul(other PartialInt8s) PartialInt8s"
  - "func (p PartialInt8s) Min(other PartialInt8s) PartialInt8s"
  - "func (p PartialInt8s) Max(other PartialInt8s) PartialInt8s"
  - "func (p PartialInt8s) Clamp(lower, upper PartialInt8s) PartialInt8s"
playUrl:
variantHelpers:
  - plugin#simd#partialint8s
  - plugin#simd#partialint16s
  - plugin#simd#partialint32s
  - plugin#simd#partialint64s
  - plugin#simd#partialuint8s
  - plugin#simd#partialuint16s
  - plugin#simd#partialuint32s
  - plugin#simd#partialuint64s
  - plugin#simd#partialfloat32s
  - plugin#simd#partialfloat64s
similarHelpers:
  - plugin#simd#vectorize
  - plugin#simd#broadcast
  - plugin#simd#contains
  - plugin#simd#select
position: 0
---

A vector whose first `Count()` lanes hold data and whose remaining lanes are padding.

A stream rarely delivers a multiple of the lane width, so the last vector of a batch is short. Rather than dropping those values or padding them into the result, a `Partial` type carries a validity mask alongside its lanes: every operation leaves padded lanes at their previous value, and nothing downstream can observe them.

There is one per element type — `PartialInt8s`, `PartialUint32s`, `PartialFloat64s` and so on.

Every element-wise operator is also available as a method. Methods chain inside `ro.Map` and need no type arguments, which is usually shorter than stacking operators:

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe3[int8, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8,
    ro.Map(func(v rosimd.PartialInt8s) []int8 {
        return v.Add(rosimd.BroadcastInt8(100)).Min(rosimd.BroadcastInt8(102)).Values()
    }),
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 101
// 102
// 102
```

`Values()` returns the valid lanes as a slice, which is how a pipeline leaves vector space — there is no devectorize operator. `Count()` reports how many lanes hold data, `Len()` the lane capacity of the running architecture, and `Sum()` folds the valid lanes to a scalar.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, string](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8,
    ro.Map(func(v rosimd.PartialInt8s) string {
        return fmt.Sprintf("%d lanes, sum %d", v.Count(), v.Sum())
    }),
)

sub := obs.Subscribe(ro.OnNext(func(line string) {
    fmt.Println(line)
}))
defer sub.Unsubscribe()

// 3 lanes, sum 6
```

Operators that need the validity mask — `Vectorize` and `ReduceContains` — accept only the `Partial` types, because `simd.Int8s` carries no mask. Every other operator accepts both.
