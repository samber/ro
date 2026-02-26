---
name: Clamp
slug: clamp
sourceRef: plugins/exp/simd/math_avx.go
type: plugin
category: simd
signatures:
  - "func ClampInt8x16[T ~int8](minValue, maxValue T)"
  - "func ClampInt16x8[T ~int16](minValue, maxValue T)"
  - "func ClampInt32x4[T ~int32](minValue, maxValue T)"
  - "func ClampInt64x2[T ~int64](minValue, maxValue T)"
  - "func ClampUint8x16[T ~uint8](minValue, maxValue T)"
  - "func ClampUint16x8[T ~uint16](minValue, maxValue T)"
  - "func ClampUint32x4[T ~uint32](minValue, maxValue T)"
  - "func ClampUint64x2[T ~uint64](minValue, maxValue T)"
  - "func ClampFloat32x4[T ~float32](minValue, maxValue T)"
  - "func ClampFloat64x2[T ~float64](minValue, maxValue T)"
  - "func ClampInt8x32[T ~int8](minValue, maxValue T)"
  - "func ClampInt16x16[T ~int16](minValue, maxValue T)"
  - "func ClampInt32x8[T ~int32](minValue, maxValue T)"
  - "func ClampInt64x4[T ~int64](minValue, maxValue T)"
  - "func ClampUint8x32[T ~uint8](minValue, maxValue T)"
  - "func ClampUint16x16[T ~uint16](minValue, maxValue T)"
  - "func ClampUint32x8[T ~uint32](minValue, maxValue T)"
  - "func ClampUint64x4[T ~uint64](minValue, maxValue T)"
  - "func ClampFloat32x8[T ~float32](minValue, maxValue T)"
  - "func ClampFloat64x4[T ~float64](minValue, maxValue T)"
  - "func ClampInt8x64[T ~int8](minValue, maxValue T)"
  - "func ClampInt16x32[T ~int16](minValue, maxValue T)"
  - "func ClampInt32x16[T ~int32](minValue, maxValue T)"
  - "func ClampInt64x8[T ~int64](minValue, maxValue T)"
  - "func ClampUint8x64[T ~uint8](minValue, maxValue T)"
  - "func ClampUint16x32[T ~uint16](minValue, maxValue T)"
  - "func ClampUint32x16[T ~uint32](minValue, maxValue T)"
  - "func ClampUint64x8[T ~uint64](minValue, maxValue T)"
  - "func ClampFloat32x16[T ~float32](minValue, maxValue T)"
  - "func ClampFloat64x8[T ~float64](minValue, maxValue T)"
playUrl:
variantHelpers:
  - plugin#simd#clampint8x16
  - plugin#simd#clampint16x8
  - plugin#simd#clampint32x4
  - plugin#simd#clampint64x2
  - plugin#simd#clampuint8x16
  - plugin#simd#clampuint16x8
  - plugin#simd#clampuint32x4
  - plugin#simd#clampuint64x2
  - plugin#simd#clampfloat32x4
  - plugin#simd#clampfloat64x2
  - plugin#simd#clampint8x32
  - plugin#simd#clampint16x16
  - plugin#simd#clampint32x8
  - plugin#simd#clampint64x4
  - plugin#simd#clampuint8x32
  - plugin#simd#clampuint16x16
  - plugin#simd#clampuint32x8
  - plugin#simd#clampuint64x4
  - plugin#simd#clampfloat32x8
  - plugin#simd#clampfloat64x4
  - plugin#simd#clampint8x64
  - plugin#simd#clampint16x32
  - plugin#simd#clampint32x16
  - plugin#simd#clampint64x8
  - plugin#simd#clampuint8x64
  - plugin#simd#clampuint16x32
  - plugin#simd#clampuint32x16
  - plugin#simd#clampuint64x8
  - plugin#simd#clampfloat32x16
  - plugin#simd#clampfloat64x8
similarHelpers:
  - plugin#simd#add
  - plugin#simd#sub
  - plugin#simd#min
  - plugin#simd#max
position: 40
---

Clamps all lanes in SIMD vectors to a specified range [minValue, maxValue] using SIMD instructions for parallel computation.

```go
import (
    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(-200), float32(-100), float32(0), float32(100),
    ),
    rosimd.ScalarToFloat32x4[float32](),
    rosimd.ClampFloat32x4[float32](-50, 50),
    rosimd.Float32x4ToScalar[float32](),
)

sub := obs.Subscribe(ro.NewObserver[float32](
    func(v float32) {
        fmt.Printf("Next: %.1f\n", v)
    },
    ro.OnError(func(err error) {
        fmt.Printf("Error: %v\n", err)
    }),
    ro.OnComplete(func() {
        fmt.Println("Completed")
    }),
))
defer sub.Unsubscribe()

// Next: -50.0
// Next: -50.0
// Next: 0.0
// Next: 50.0
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- ClampFloat32x4
- ClampFloat64x2
- ClampInt8x16
- ClampInt16x8
- ClampInt32x4
- ClampInt64x2
- ClampUint8x16
- ClampUint16x8
- ClampUint32x4
- ClampUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- ClampFloat32x8
- ClampFloat64x4
- ClampInt8x32
- ClampInt16x16
- ClampInt32x8
- ClampUint8x32
- ClampUint16x16
- ClampUint32x8

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- ClampFloat32x16
- ClampFloat64x8
- ClampInt8x64
- ClampInt16x32
- ClampInt32x16
- ClampInt64x4 (256-bit; requires AVX-512 for int64 min/max)
- ClampInt64x8
- ClampUint8x64
- ClampUint16x32
- ClampUint32x16
- ClampUint64x4 (256-bit; requires AVX-512 for uint64 min/max)
- ClampUint64x8
