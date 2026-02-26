---
name: ScalarToSIMD
slug: scalartosimd
sourceRef: plugins/exp/simd/conversion_avx.go
type: plugin
category: simd
signatures:
  - "func ScalarToInt8x16[T ~int8]()"
  - "func ScalarToInt16x8[T ~int16]()"
  - "func ScalarToInt32x4[T ~int32]()"
  - "func ScalarToInt64x2[T ~int64]()"
  - "func ScalarToUint8x16[T ~uint8]()"
  - "func ScalarToUint16x8[T ~uint16]()"
  - "func ScalarToUint32x4[T ~uint32]()"
  - "func ScalarToUint64x2[T ~uint64]()"
  - "func ScalarToFloat32x4[T ~float32]()"
  - "func ScalarToFloat64x2[T ~float64]()"
  - "func ScalarToInt8x32[T ~int8]()"
  - "func ScalarToInt16x16[T ~int16]()"
  - "func ScalarToInt32x8[T ~int32]()"
  - "func ScalarToInt64x4[T ~int64]()"
  - "func ScalarToUint8x32[T ~uint8]()"
  - "func ScalarToUint16x16[T ~uint16]()"
  - "func ScalarToUint32x8[T ~uint32]()"
  - "func ScalarToUint64x4[T ~uint64]()"
  - "func ScalarToFloat32x8[T ~float32]()"
  - "func ScalarToFloat64x4[T ~float64]()"
  - "func ScalarToInt8x64[T ~int8]()"
  - "func ScalarToInt16x32[T ~int16]()"
  - "func ScalarToInt32x16[T ~int32]()"
  - "func ScalarToInt64x8[T ~int64]()"
  - "func ScalarToUint8x64[T ~uint8]()"
  - "func ScalarToUint16x32[T ~uint16]()"
  - "func ScalarToUint32x16[T ~uint32]()"
  - "func ScalarToUint64x8[T ~uint64]()"
  - "func ScalarToFloat32x16[T ~float32]()"
  - "func ScalarToFloat64x8[T ~float64]()"
playUrl:
variantHelpers:
  - plugin#simd#scalartoint8x16
  - plugin#simd#scalartoint16x8
  - plugin#simd#scalartoint32x4
  - plugin#simd#scalartoint64x2
  - plugin#simd#scalartouint8x16
  - plugin#simd#scalartouint16x8
  - plugin#simd#scalartouint32x4
  - plugin#simd#scalartouint64x2
  - plugin#simd#scalartofloat32x4
  - plugin#simd#scalartofloat64x2
  - plugin#simd#scalartoint8x32
  - plugin#simd#scalartoint16x16
  - plugin#simd#scalartoint32x8
  - plugin#simd#scalartoint64x4
  - plugin#simd#scalartouint8x32
  - plugin#simd#scalartouint16x16
  - plugin#simd#scalartouint32x8
  - plugin#simd#scalartouint64x4
  - plugin#simd#scalartofloat32x8
  - plugin#simd#scalartofloat64x4
  - plugin#simd#scalartoint8x64
  - plugin#simd#scalartoint16x32
  - plugin#simd#scalartoint32x16
  - plugin#simd#scalartoint64x8
  - plugin#simd#scalartouint8x64
  - plugin#simd#scalartouint16x32
  - plugin#simd#scalartouint32x16
  - plugin#simd#scalartouint64x8
  - plugin#simd#scalartofloat32x16
  - plugin#simd#scalartofloat64x8
similarHelpers:
  - plugin#simd#simdtoscalar
position: 0
---

Converts streams of scalar values into SIMD vectors. Each variant buffers a specific number of scalar values (based on the vector size: 2, 4, 8, 16, 32, or 64) and emits them as a single SIMD vector.

```go
import (
    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[int8, *archsimd.Int8x16](
    ro.Just(
        int8(1), int8(2), int8(3), int8(4),
        int8(5), int8(6), int8(7), int8(8),
        int8(9), int8(10), int8(11), int8(12),
        int8(13), int8(14), int8(15), int8(16),
        int8(17), int8(18), // partial final vector
    ),
    rosimd.ScalarToInt8x16[int8](),
)

sub := obs.Subscribe(ro.NewObserver[*archsimd.Int8x16](
    func(vec *archsimd.Int8x16) {
        var buf [16]int8
        vec.Store(&buf)
        fmt.Printf("Next: %v\n", buf[:])
    },
    ro.OnError(func(err error) {
        fmt.Printf("Error: %v\n", err)
    }),
    ro.OnComplete(func() {
        fmt.Println("Completed")
    }),
))
defer sub.Unsubscribe()

// Next: [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16]
// Next: [17 18 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- ScalarToInt8x16
- ScalarToInt16x8
- ScalarToInt32x4
- ScalarToInt64x2
- ScalarToUint8x16
- ScalarToUint16x8
- ScalarToUint32x4
- ScalarToUint64x2
- ScalarToFloat32x4
- ScalarToFloat64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- ScalarToInt8x32
- ScalarToInt16x16
- ScalarToInt32x8
- ScalarToInt64x4
- ScalarToUint8x32
- ScalarToUint16x16
- ScalarToUint32x8
- ScalarToUint64x4
- ScalarToFloat32x8
- ScalarToFloat64x4

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- ScalarToInt8x64
- ScalarToInt16x32
- ScalarToInt32x16
- ScalarToInt64x8
- ScalarToUint8x64
- ScalarToUint16x32
- ScalarToUint32x16
- ScalarToUint64x8
- ScalarToFloat32x16
- ScalarToFloat64x8
