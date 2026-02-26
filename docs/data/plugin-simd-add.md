---
name: Add
slug: add
sourceRef: plugins/exp/simd/math_avx.go
type: plugin
category: simd
signatures:
  - "func AddInt8x16[T ~int8](number T)"
  - "func AddInt16x8[T ~int16](number T)"
  - "func AddInt32x4[T ~int32](number T)"
  - "func AddInt64x2[T ~int64](number T)"
  - "func AddUint8x16[T ~uint8](number T)"
  - "func AddUint16x8[T ~uint16](number T)"
  - "func AddUint32x4[T ~uint32](number T)"
  - "func AddUint64x2[T ~uint64](number T)"
  - "func AddFloat32x4[T ~float32](number T)"
  - "func AddFloat64x2[T ~float64](number T)"
  - "func AddInt8x32[T ~int8](number T)"
  - "func AddInt16x16[T ~int16](number T)"
  - "func AddInt32x8[T ~int32](number T)"
  - "func AddInt64x4[T ~int64](number T)"
  - "func AddUint8x32[T ~uint8](number T)"
  - "func AddUint16x16[T ~uint16](number T)"
  - "func AddUint32x8[T ~uint32](number T)"
  - "func AddUint64x4[T ~uint64](number T)"
  - "func AddFloat32x8[T ~float32](number T)"
  - "func AddFloat64x4[T ~float64](number T)"
  - "func AddInt8x64[T ~int8](number T)"
  - "func AddInt16x32[T ~int16](number T)"
  - "func AddInt32x16[T ~int32](number T)"
  - "func AddInt64x8[T ~int64](number T)"
  - "func AddUint8x64[T ~uint8](number T)"
  - "func AddUint16x32[T ~uint16](number T)"
  - "func AddUint32x16[T ~uint32](number T)"
  - "func AddUint64x8[T ~uint64](number T)"
  - "func AddFloat32x16[T ~float32](number T)"
  - "func AddFloat64x8[T ~float64](number T)"
playUrl:
variantHelpers:
  - plugin#simd#addint8x16
  - plugin#simd#addint16x8
  - plugin#simd#addint32x4
  - plugin#simd#addint64x2
  - plugin#simd#adduint8x16
  - plugin#simd#adduint16x8
  - plugin#simd#adduint32x4
  - plugin#simd#adduint64x2
  - plugin#simd#addfloat32x4
  - plugin#simd#addfloat64x2
  - plugin#simd#addint8x32
  - plugin#simd#addint16x16
  - plugin#simd#addint32x8
  - plugin#simd#addint64x4
  - plugin#simd#adduint8x32
  - plugin#simd#adduint16x16
  - plugin#simd#adduint32x8
  - plugin#simd#adduint64x4
  - plugin#simd#addfloat32x8
  - plugin#simd#addfloat64x4
  - plugin#simd#addint8x64
  - plugin#simd#addint16x32
  - plugin#simd#addint32x16
  - plugin#simd#addint64x8
  - plugin#simd#adduint8x64
  - plugin#simd#adduint16x32
  - plugin#simd#adduint32x16
  - plugin#simd#adduint64x8
  - plugin#simd#addfloat32x16
  - plugin#simd#addfloat64x8
similarHelpers:
  - plugin#simd#sub
  - plugin#simd#clamp
  - plugin#simd#min
  - plugin#simd#max
position: 20
---

Adds a scalar number to all lanes in SIMD vectors using SIMD instructions for parallel computation.

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
    rosimd.AddFloat32x4[float32](100),
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

// Next: 110.0
// Next: 120.0
// Next: 130.0
// Next: 140.0
// Next: 105.0
// Next: 110.0
// Next: 115.0
// Next: 120.0
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- AddFloat32x4
- AddFloat64x2
- AddInt8x16
- AddInt16x8
- AddInt32x4
- AddInt64x2
- AddUint8x16
- AddUint16x8
- AddUint32x4
- AddUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- AddFloat32x8
- AddFloat64x4
- AddInt8x32
- AddInt16x16
- AddInt32x8
- AddInt64x4
- AddUint8x32
- AddUint16x16
- AddUint32x8
- AddUint64x4

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- AddFloat32x16
- AddFloat64x8
- AddInt8x64
- AddInt16x32
- AddInt32x16
- AddInt64x8
- AddUint8x64
- AddUint16x32
- AddUint32x16
- AddUint64x8
