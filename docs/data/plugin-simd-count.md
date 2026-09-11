---
name: Count
slug: count
sourceRef: plugins/exp/simd/vectorize.go#L230
type: plugin
category: simd
signatures:
  - "func Count[V LaneCount]()"
playUrl:
variantHelpers:
  - plugin#simd#count
similarHelpers:
  - plugin#simd#toscalar
  - plugin#simd#flatten
  - plugin#simd#vectorize
position: 48
---

Reports how many lanes of each vector hold data.

It emits one value per vector rather than one per lane, so it describes the batching rather than the data: every vector counts a full register except the last of a stream, which is short whenever the stream length is not a multiple of the lane width.

```go
import (
    "fmt"
    "simd"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

lanes := simd.BroadcastInt8s(0).Len()

// One and a half registers' worth, so the second batch is half full.
input := make([]int8, lanes+lanes/2)

obs := ro.Pipe2[int8, rosimd.PartialInt8s, int](
    ro.FromSlice(input),
    rosimd.VectorizeInt8[rosimd.PartialInt8s](),
    rosimd.Count[rosimd.PartialInt8s](),
)

sub := obs.Subscribe(ro.OnNext(func(count int) {
    fmt.Println(count)
}))
defer sub.Unsubscribe()

// on a 128-bit machine, where int8 has 16 lanes:
// 16
// 8
```

One operator serves every element type. Unlike `ToScalar` and `Flatten` it rejects the standard library's vector types, which carry no validity mask and so have no count distinct from their capacity.
