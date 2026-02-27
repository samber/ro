---
name: Max
slug: max
sourceRef: plugins/exp/simd/math_avx.go
type: plugin
category: simd
signatures:
  - "func MaxInt8x16[T ~int8](maxValue T)"
  - "func MaxInt16x8[T ~int16](maxValue T)"
  - "func MaxInt32x4[T ~int32](maxValue T)"
  - "func MaxInt64x2[T ~int64](maxValue T)"
  - "func MaxUint8x16[T ~uint8](maxValue T)"
  - "func MaxUint16x8[T ~uint16](maxValue T)"
  - "func MaxUint32x4[T ~uint32](maxValue T)"
  - "func MaxUint64x2[T ~uint64](maxValue T)"
  - "func MaxFloat32x4[T ~float32](maxValue T)"
  - "func MaxFloat64x2[T ~float64](maxValue T)"
  - "func MaxInt8x32[T ~int8](maxValue T)"
  - "func MaxInt16x16[T ~int16](maxValue T)"
  - "func MaxInt32x8[T ~int32](maxValue T)"
  - "func MaxInt64x4[T ~int64](maxValue T)"
  - "func MaxUint8x32[T ~uint8](maxValue T)"
  - "func MaxUint16x16[T ~uint16](maxValue T)"
  - "func MaxUint32x8[T ~uint32](maxValue T)"
  - "func MaxUint64x4[T ~uint64](maxValue T)"
  - "func MaxFloat32x8[T ~float32](maxValue T)"
  - "func MaxFloat64x4[T ~float64](maxValue T)"
  - "func MaxInt8x64[T ~int8](maxValue T)"
  - "func MaxInt16x32[T ~int16](maxValue T)"
  - "func MaxInt32x16[T ~int32](maxValue T)"
  - "func MaxInt64x8[T ~int64](maxValue T)"
  - "func MaxUint8x64[T ~uint8](maxValue T)"
  - "func MaxUint16x32[T ~uint16](maxValue T)"
  - "func MaxUint32x16[T ~uint32](maxValue T)"
  - "func MaxUint64x8[T ~uint64](maxValue T)"
  - "func MaxFloat32x16[T ~float32](maxValue T)"
  - "func MaxFloat64x8[T ~float64](maxValue T)"
playUrl:
variantHelpers:
  - plugin#simd#maxint8x16
  - plugin#simd#maxint16x8
  - plugin#simd#maxint32x4
  - plugin#simd#maxint64x2
  - plugin#simd#maxuint8x16
  - plugin#simd#maxuint16x8
  - plugin#simd#maxuint32x4
  - plugin#simd#maxuint64x2
  - plugin#simd#maxfloat32x4
  - plugin#simd#maxfloat64x2
  - plugin#simd#maxint8x32
  - plugin#simd#maxint16x16
  - plugin#simd#maxint32x8
  - plugin#simd#maxint64x4
  - plugin#simd#maxuint8x32
  - plugin#simd#maxuint16x16
  - plugin#simd#maxuint32x8
  - plugin#simd#maxuint64x4
  - plugin#simd#maxfloat32x8
  - plugin#simd#maxfloat64x4
  - plugin#simd#maxint8x64
  - plugin#simd#maxint16x32
  - plugin#simd#maxint32x16
  - plugin#simd#maxint64x8
  - plugin#simd#maxuint8x64
  - plugin#simd#maxuint16x32
  - plugin#simd#maxuint32x16
  - plugin#simd#maxuint64x8
  - plugin#simd#maxfloat32x16
  - plugin#simd#maxfloat64x8
similarHelpers:
  - plugin#simd#add
  - plugin#simd#sub
  - plugin#simd#clamp
  - plugin#simd#min
position: 60
---

Ensures all lanes in SIMD vectors are at most the specified maximum value using SIMD instructions for parallel computation. Values above the maximum are replaced with the maximum.

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
    rosimd.MaxFloat32x4[float32](50),
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

// Next: -200.0
// Next: -100.0
// Next: 0.0
// Next: 50.0
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- MaxFloat32x4
- MaxFloat64x2
- MaxInt8x16
- MaxInt16x8
- MaxInt32x4
- MaxInt64x2
- MaxUint8x16
- MaxUint16x8
- MaxUint32x4
- MaxUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- MaxFloat32x8
- MaxFloat64x4
- MaxInt8x32
- MaxInt16x16
- MaxInt32x8
- MaxUint8x32
- MaxUint16x16
- MaxUint32x8

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- MaxFloat32x16
- MaxFloat64x8
- MaxInt8x64
- MaxInt16x32
- MaxInt32x16
- MaxInt64x4 (256-bit; requires AVX-512 for int64 max)
- MaxInt64x8
- MaxUint8x64
- MaxUint16x32
- MaxUint32x16
- MaxUint64x4 (256-bit; requires AVX-512 for uint64 max)
- MaxUint64x8
