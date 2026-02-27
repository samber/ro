---
name: Min
slug: min
sourceRef: plugins/exp/simd/math_avx.go
type: plugin
category: simd
signatures:
  - "func MinInt8x16[T ~int8](minValue T)"
  - "func MinInt16x8[T ~int16](minValue T)"
  - "func MinInt32x4[T ~int32](minValue T)"
  - "func MinInt64x2[T ~int64](minValue T)"
  - "func MinUint8x16[T ~uint8](minValue T)"
  - "func MinUint16x8[T ~uint16](minValue T)"
  - "func MinUint32x4[T ~uint32](minValue T)"
  - "func MinUint64x2[T ~uint64](minValue T)"
  - "func MinFloat32x4[T ~float32](minValue T)"
  - "func MinFloat64x2[T ~float64](minValue T)"
  - "func MinInt8x32[T ~int8](minValue T)"
  - "func MinInt16x16[T ~int16](minValue T)"
  - "func MinInt32x8[T ~int32](minValue T)"
  - "func MinInt64x4[T ~int64](minValue T)"
  - "func MinUint8x32[T ~uint8](minValue T)"
  - "func MinUint16x16[T ~uint16](minValue T)"
  - "func MinUint32x8[T ~uint32](minValue T)"
  - "func MinUint64x4[T ~uint64](minValue T)"
  - "func MinFloat32x8[T ~float32](minValue T)"
  - "func MinFloat64x4[T ~float64](minValue T)"
  - "func MinInt8x64[T ~int8](minValue T)"
  - "func MinInt16x32[T ~int16](minValue T)"
  - "func MinInt32x16[T ~int32](minValue T)"
  - "func MinInt64x8[T ~int64](minValue T)"
  - "func MinUint8x64[T ~uint8](minValue T)"
  - "func MinUint16x32[T ~uint16](minValue T)"
  - "func MinUint32x16[T ~uint32](minValue T)"
  - "func MinUint64x8[T ~uint64](minValue T)"
  - "func MinFloat32x16[T ~float32](minValue T)"
  - "func MinFloat64x8[T ~float64](minValue T)"
playUrl:
variantHelpers:
  - plugin#simd#minint8x16
  - plugin#simd#minint16x8
  - plugin#simd#minint32x4
  - plugin#simd#minint64x2
  - plugin#simd#minuint8x16
  - plugin#simd#minuint16x8
  - plugin#simd#minuint32x4
  - plugin#simd#minuint64x2
  - plugin#simd#minfloat32x4
  - plugin#simd#minfloat64x2
  - plugin#simd#minint8x32
  - plugin#simd#minint16x16
  - plugin#simd#minint32x8
  - plugin#simd#minint64x4
  - plugin#simd#minuint8x32
  - plugin#simd#minuint16x16
  - plugin#simd#minuint32x8
  - plugin#simd#minuint64x4
  - plugin#simd#minfloat32x8
  - plugin#simd#minfloat64x4
  - plugin#simd#minint8x64
  - plugin#simd#minint16x32
  - plugin#simd#minint32x16
  - plugin#simd#minint64x8
  - plugin#simd#minuint8x64
  - plugin#simd#minuint16x32
  - plugin#simd#minuint32x16
  - plugin#simd#minuint64x8
  - plugin#simd#minfloat32x16
  - plugin#simd#minfloat64x8
similarHelpers:
  - plugin#simd#add
  - plugin#simd#sub
  - plugin#simd#clamp
  - plugin#simd#max
position: 50
---

Ensures all lanes in SIMD vectors are at least the specified minimum value using SIMD instructions for parallel computation. Values below the minimum are replaced with the minimum.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(-200), float32(-100), float32(0), float32(100),
    ),
    rosimd.ScalarToFloat32x4[float32](),
    rosimd.MinFloat32x4[float32](-50),
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
// Next: 100.0
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- MinFloat32x4
- MinFloat64x2
- MinInt8x16
- MinInt16x8
- MinInt32x4
- MinInt64x2
- MinUint8x16
- MinUint16x8
- MinUint32x4
- MinUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- MinFloat32x8
- MinFloat64x4
- MinInt8x32
- MinInt16x16
- MinInt32x8
- MinUint8x32
- MinUint16x16
- MinUint32x8

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- MinFloat32x16
- MinFloat64x8
- MinInt8x64
- MinInt16x32
- MinInt32x16
- MinInt64x4 (256-bit; requires AVX-512 for int64 min)
- MinInt64x8
- MinUint8x64
- MinUint16x32
- MinUint32x16
- MinUint64x4 (256-bit; requires AVX-512 for uint64 min)
- MinUint64x8
