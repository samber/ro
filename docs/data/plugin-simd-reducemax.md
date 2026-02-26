---
name: ReduceMax
slug: reducemax
sourceRef: plugins/exp/simd/math_sse.go
type: plugin
category: simd
signatures:
  - "func ReduceMaxInt8x16[T ~int8]()"
  - "func ReduceMaxInt16x8[T ~int16]()"
  - "func ReduceMaxInt32x4[T ~int32]()"
  - "func ReduceMaxInt64x2[T ~int64]()"
  - "func ReduceMaxUint8x16[T ~uint8]()"
  - "func ReduceMaxUint16x8[T ~uint16]()"
  - "func ReduceMaxUint32x4[T ~uint32]()"
  - "func ReduceMaxUint64x2[T ~uint64]()"
  - "func ReduceMaxFloat32x4[T ~float32]()"
  - "func ReduceMaxFloat64x2[T ~float64]()"
  - "func ReduceMaxInt8x32[T ~int8]()"
  - "func ReduceMaxInt16x16[T ~int16]()"
  - "func ReduceMaxInt32x8[T ~int32]()"
  - "func ReduceMaxInt64x4[T ~int64]()"
  - "func ReduceMaxUint8x32[T ~uint8]()"
  - "func ReduceMaxUint16x16[T ~uint16]()"
  - "func ReduceMaxUint32x8[T ~uint32]()"
  - "func ReduceMaxUint64x4[T ~uint64]()"
  - "func ReduceMaxFloat32x8[T ~float32]()"
  - "func ReduceMaxFloat64x4[T ~float64]()"
  - "func ReduceMaxInt8x64[T ~int8]()"
  - "func ReduceMaxInt16x32[T ~int16]()"
  - "func ReduceMaxInt32x16[T ~int32]()"
  - "func ReduceMaxInt64x8[T ~int64]()"
  - "func ReduceMaxUint8x64[T ~uint8]()"
  - "func ReduceMaxUint16x32[T ~uint16]()"
  - "func ReduceMaxUint32x16[T ~uint32]()"
  - "func ReduceMaxUint64x8[T ~uint64]()"
  - "func ReduceMaxFloat32x16[T ~float32]()"
  - "func ReduceMaxFloat64x8[T ~float64]()"
playUrl:
variantHelpers:
  - plugin#simd#reducemaxint8x16
  - plugin#simd#reducemaxint16x8
  - plugin#simd#reducemaxint32x4
  - plugin#simd#reducemaxint64x2
  - plugin#simd#reducemaxuint8x16
  - plugin#simd#reducemaxuint16x8
  - plugin#simd#reducemaxuint32x4
  - plugin#simd#reducemaxuint64x2
  - plugin#simd#reducemaxfloat32x4
  - plugin#simd#reducemaxfloat64x2
  - plugin#simd#reducemaxint8x32
  - plugin#simd#reducemaxint16x16
  - plugin#simd#reducemaxint32x8
  - plugin#simd#reducemaxint64x4
  - plugin#simd#reducemaxuint8x32
  - plugin#simd#reducemaxuint16x16
  - plugin#simd#reducemaxuint32x8
  - plugin#simd#reducemaxuint64x4
  - plugin#simd#reducemaxfloat32x8
  - plugin#simd#reducemaxfloat64x4
  - plugin#simd#reducemaxint8x64
  - plugin#simd#reducemaxint16x32
  - plugin#simd#reducemaxint32x16
  - plugin#simd#reducemaxint64x8
  - plugin#simd#reducemaxuint8x64
  - plugin#simd#reducemaxuint16x32
  - plugin#simd#reducemaxuint32x16
  - plugin#simd#reducemaxuint64x8
  - plugin#simd#reducemaxfloat32x16
  - plugin#simd#reducemaxfloat64x8
similarHelpers:
  - plugin#simd#reducesum
  - plugin#simd#reducemin
position: 90
---

Finds the maximum value across all lanes of SIMD vectors and emits a single scalar value when the source completes.

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
    rosimd.ReduceMaxFloat32x4[float32](),
)

sub := obs.Subscribe(ro.NewObserver[float32](
    func(max float32) {
        fmt.Printf("Next: %.1f\n", max)
    },
    ro.OnError(func(err error) {
        fmt.Printf("Error: %v\n", err)
    }),
    ro.OnComplete(func() {
        fmt.Println("Completed")
    }),
))
defer sub.Unsubscribe()

// Next: 40.0
// Completed
```

## SSE variants (128-bit vectors)

Available on all x86_64 CPUs with SSE support (basically all modern x86_64 CPUs).

- ReduceMaxFloat32x4
- ReduceMaxFloat64x2
- ReduceMaxInt8x16
- ReduceMaxInt16x8
- ReduceMaxInt32x4
- ReduceMaxInt64x2
- ReduceMaxUint8x16
- ReduceMaxUint16x8
- ReduceMaxUint32x4
- ReduceMaxUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- ReduceMaxFloat32x8
- ReduceMaxFloat64x4
- ReduceMaxInt8x32
- ReduceMaxInt16x16
- ReduceMaxInt32x8
- ReduceMaxUint8x32
- ReduceMaxUint16x16
- ReduceMaxUint32x8

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- ReduceMaxFloat32x16
- ReduceMaxFloat64x8
- ReduceMaxInt8x64
- ReduceMaxInt16x32
- ReduceMaxInt32x16
- ReduceMaxInt64x4 (256-bit; requires AVX-512 for int64 max)
- ReduceMaxInt64x8
- ReduceMaxUint8x64
- ReduceMaxUint16x32
- ReduceMaxUint32x16
- ReduceMaxUint64x4 (256-bit; requires AVX-512 for uint64 max)
- ReduceMaxUint64x8
