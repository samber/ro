---
name: Broadcast
slug: broadcast
sourceRef: plugins/exp/simd/int8.go#L67
type: plugin
category: simd
signatures:
  - "func BroadcastInt8(value int8) PartialInt8s"
  - "func BroadcastInt16(value int16) PartialInt16s"
  - "func BroadcastInt32(value int32) PartialInt32s"
  - "func BroadcastInt64(value int64) PartialInt64s"
  - "func BroadcastUint8(value uint8) PartialUint8s"
  - "func BroadcastUint16(value uint16) PartialUint16s"
  - "func BroadcastUint32(value uint32) PartialUint32s"
  - "func BroadcastUint64(value uint64) PartialUint64s"
  - "func BroadcastFloat32(value float32) PartialFloat32s"
  - "func BroadcastFloat64(value float64) PartialFloat64s"
playUrl:
variantHelpers:
  - plugin#simd#broadcastint8
  - plugin#simd#broadcastint16
  - plugin#simd#broadcastint32
  - plugin#simd#broadcastint64
  - plugin#simd#broadcastuint8
  - plugin#simd#broadcastuint16
  - plugin#simd#broadcastuint32
  - plugin#simd#broadcastuint64
  - plugin#simd#broadcastfloat32
  - plugin#simd#broadcastfloat64
similarHelpers:
  - plugin#simd#partial
  - plugin#simd#add
  - plugin#simd#clamp
position: 10
---

Returns a full vector holding `value` in every lane.

SIMD has no scalar-operand arithmetic: `Add` takes another vector. `Broadcast` is how a scalar becomes one, and every element-wise operator expects its operand to have been widened this way.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe4[int8, rosimd.PartialInt8s, rosimd.PartialInt8s, []int8, int8](
    ro.Just[int8](1, 2, 3),
    rosimd.VectorizeInt8,
    rosimd.AddInt8(rosimd.BroadcastInt8(10)),
    rosimd.ToScalar,
    ro.Flatten[int8](),
)

sub := obs.Subscribe(ro.OnNext(func(value int8) {
    fmt.Println(value)
}))
defer sub.Unsubscribe()

// 11
// 12
// 13
```

Widening at the call site rather than inside the operator is also what makes the type argument inferable, so call sites stay free of `[rosimd.PartialInt8s]`.
