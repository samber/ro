---
name: Sub
slug: sub
sourceRef: plugins/exp/simd/math_avx.go
type: plugin
category: simd
signatures:
  - "func SubInt8x16[T ~int8](number T)"
  - "func SubInt16x8[T ~int16](number T)"
  - "func SubInt32x4[T ~int32](number T)"
  - "func SubInt64x2[T ~int64](number T)"
  - "func SubUint8x16[T ~uint8](number T)"
  - "func SubUint16x8[T ~uint16](number T)"
  - "func SubUint32x4[T ~uint32](number T)"
  - "func SubUint64x2[T ~uint64](number T)"
  - "func SubFloat32x4[T ~float32](number T)"
  - "func SubFloat64x2[T ~float64](number T)"
  - "func SubInt8x32[T ~int8](number T)"
  - "func SubInt16x16[T ~int16](number T)"
  - "func SubInt32x8[T ~int32](number T)"
  - "func SubInt64x4[T ~int64](number T)"
  - "func SubUint8x32[T ~uint8](number T)"
  - "func SubUint16x16[T ~uint16](number T)"
  - "func SubUint32x8[T ~uint32](number T)"
  - "func SubUint64x4[T ~uint64](number T)"
  - "func SubFloat32x8[T ~float32](number T)"
  - "func SubFloat64x4[T ~float64](number T)"
  - "func SubInt8x64[T ~int8](number T)"
  - "func SubInt16x32[T ~int16](number T)"
  - "func SubInt32x16[T ~int32](number T)"
  - "func SubInt64x8[T ~int64](number T)"
  - "func SubUint8x64[T ~uint8](number T)"
  - "func SubUint16x32[T ~uint16](number T)"
  - "func SubUint32x16[T ~uint32](number T)"
  - "func SubUint64x8[T ~uint64](number T)"
  - "func SubFloat32x16[T ~float32](number T)"
  - "func SubFloat64x8[T ~float64](number T)"
playUrl:
variantHelpers:
  - plugin#simd#subint8x16
  - plugin#simd#subint16x8
  - plugin#simd#subint32x4
  - plugin#simd#subint64x2
  - plugin#simd#subuint8x16
  - plugin#simd#subuint16x8
  - plugin#simd#subuint32x4
  - plugin#simd#subuint64x2
  - plugin#simd#subfloat32x4
  - plugin#simd#subfloat64x2
  - plugin#simd#subint8x32
  - plugin#simd#subint16x16
  - plugin#simd#subint32x8
  - plugin#simd#subint64x4
  - plugin#simd#subuint8x32
  - plugin#simd#subuint16x16
  - plugin#simd#subuint32x8
  - plugin#simd#subuint64x4
  - plugin#simd#subfloat32x8
  - plugin#simd#subfloat64x4
  - plugin#simd#subint8x64
  - plugin#simd#subint16x32
  - plugin#simd#subint32x16
  - plugin#simd#subint64x8
  - plugin#simd#subuint8x64
  - plugin#simd#subuint16x32
  - plugin#simd#subuint32x16
  - plugin#simd#subuint64x8
  - plugin#simd#subfloat32x16
  - plugin#simd#subfloat64x8
similarHelpers:
  - plugin#simd#add
  - plugin#simd#clamp
  - plugin#simd#min
  - plugin#simd#max
position: 30
---

Subtracts a scalar number from all lanes in SIMD vectors using SIMD instructions for parallel computation.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(100), float32(200), float32(300), float32(400),
    ),
    rosimd.ScalarToFloat32x4[float32](),
    rosimd.SubFloat32x4[float32](50),
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

// Next: 50.0
// Next: 150.0
// Next: 250.0
// Next: 350.0
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- SubFloat32x4
- SubFloat64x2
- SubInt8x16
- SubInt16x8
- SubInt32x4
- SubInt64x2
- SubUint8x16
- SubUint16x8
- SubUint32x4
- SubUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- SubFloat32x8
- SubFloat64x4
- SubInt8x32
- SubInt16x16
- SubInt32x8
- SubInt64x4
- SubUint8x32
- SubUint16x16
- SubUint32x8
- SubUint64x4

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- SubFloat32x16
- SubFloat64x8
- SubInt8x64
- SubInt16x32
- SubInt32x16
- SubInt64x8
- SubUint8x64
- SubUint16x32
- SubUint32x16
- SubUint64x8
