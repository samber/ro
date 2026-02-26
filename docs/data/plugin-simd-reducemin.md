---
name: ReduceMin
slug: reducemin
sourceRef: plugins/exp/simd/math_sse.go
type: plugin
category: simd
signatures:
  - "func ReduceMinInt8x16[T ~int8]()"
  - "func ReduceMinInt16x8[T ~int16]()"
  - "func ReduceMinInt32x4[T ~int32]()"
  - "func ReduceMinInt64x2[T ~int64]()"
  - "func ReduceMinUint8x16[T ~uint8]()"
  - "func ReduceMinUint16x8[T ~uint16]()"
  - "func ReduceMinUint32x4[T ~uint32]()"
  - "func ReduceMinUint64x2[T ~uint64]()"
  - "func ReduceMinFloat32x4[T ~float32]()"
  - "func ReduceMinFloat64x2[T ~float64]()"
  - "func ReduceMinInt8x32[T ~int8]()"
  - "func ReduceMinInt16x16[T ~int16]()"
  - "func ReduceMinInt32x8[T ~int32]()"
  - "func ReduceMinInt64x4[T ~int64]()"
  - "func ReduceMinUint8x32[T ~uint8]()"
  - "func ReduceMinUint16x16[T ~uint16]()"
  - "func ReduceMinUint32x8[T ~uint32]()"
  - "func ReduceMinUint64x4[T ~uint64]()"
  - "func ReduceMinFloat32x8[T ~float32]()"
  - "func ReduceMinFloat64x4[T ~float64]()"
  - "func ReduceMinInt8x64[T ~int8]()"
  - "func ReduceMinInt16x32[T ~int16]()"
  - "func ReduceMinInt32x16[T ~int32]()"
  - "func ReduceMinInt64x8[T ~int64]()"
  - "func ReduceMinUint8x64[T ~uint8]()"
  - "func ReduceMinUint16x32[T ~uint16]()"
  - "func ReduceMinUint32x16[T ~uint32]()"
  - "func ReduceMinUint64x8[T ~uint64]()"
  - "func ReduceMinFloat32x16[T ~float32]()"
  - "func ReduceMinFloat64x8[T ~float64]()"
playUrl:
variantHelpers:
  - plugin#simd#reduceminint8x16
  - plugin#simd#reduceminint16x8
  - plugin#simd#reduceminint32x4
  - plugin#simd#reduceminint64x2
  - plugin#simd#reduceminuint8x16
  - plugin#simd#reduceminuint16x8
  - plugin#simd#reduceminuint32x4
  - plugin#simd#reduceminuint64x2
  - plugin#simd#reduceminfloat32x4
  - plugin#simd#reduceminfloat64x2
  - plugin#simd#reduceminint8x32
  - plugin#simd#reduceminint16x16
  - plugin#simd#reduceminint32x8
  - plugin#simd#reduceminint64x4
  - plugin#simd#reduceminuint8x32
  - plugin#simd#reduceminuint16x16
  - plugin#simd#reduceminuint32x8
  - plugin#simd#reduceminuint64x4
  - plugin#simd#reduceminfloat32x8
  - plugin#simd#reduceminfloat64x4
  - plugin#simd#reduceminint8x64
  - plugin#simd#reduceminint16x32
  - plugin#simd#reduceminint32x16
  - plugin#simd#reduceminint64x8
  - plugin#simd#reduceminuint8x64
  - plugin#simd#reduceminuint16x32
  - plugin#simd#reduceminuint32x16
  - plugin#simd#reduceminuint64x8
  - plugin#simd#reduceminfloat32x16
  - plugin#simd#reduceminfloat64x8
similarHelpers:
  - plugin#simd#reducesum
  - plugin#simd#reducemax
position: 80
---

Finds the minimum value across all lanes of SIMD vectors and emits a single scalar value when the source completes.

```go
import (
    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(10), float32(20), float32(30), float32(40),
        float32(5), float32(10), float32(15), float32(20),
    ),
    rosimd.ScalarToFloat32x4[float32](),
    rosimd.ReduceMinFloat32x4[float32](),
)

sub := obs.Subscribe(ro.NewObserver[float32](
    func(min float32) {
        fmt.Printf("Next: %.1f\n", min)
    },
    ro.OnError(func(err error) {
        fmt.Printf("Error: %v\n", err)
    }),
    ro.OnComplete(func() {
        fmt.Println("Completed")
    }),
))
defer sub.Unsubscribe()

// Next: 5.0
// Completed
```

## SSE variants (128-bit vectors)

Available on all x86_64 CPUs with SSE support (basically all modern x86_64 CPUs).

- ReduceMinFloat32x4
- ReduceMinFloat64x2
- ReduceMinInt8x16
- ReduceMinInt16x8
- ReduceMinInt32x4
- ReduceMinInt64x2
- ReduceMinUint8x16
- ReduceMinUint16x8
- ReduceMinUint32x4
- ReduceMinUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- ReduceMinFloat32x8
- ReduceMinFloat64x4
- ReduceMinInt8x32
- ReduceMinInt16x16
- ReduceMinInt32x8
- ReduceMinUint8x32
- ReduceMinUint16x16
- ReduceMinUint32x8

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- ReduceMinFloat32x16
- ReduceMinFloat64x8
- ReduceMinInt8x64
- ReduceMinInt16x32
- ReduceMinInt32x16
- ReduceMinInt64x4 (256-bit; requires AVX-512 for int64 min)
- ReduceMinInt64x8
- ReduceMinUint8x64
- ReduceMinUint16x32
- ReduceMinUint32x16
- ReduceMinUint64x4 (256-bit; requires AVX-512 for uint64 min)
- ReduceMinUint64x8
