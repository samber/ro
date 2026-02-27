---
name: SIMDToScalar
slug: simdtoscalar
sourceRef: plugins/exp/simd/conversion_avx.go
type: plugin
category: simd
signatures:
  - "func Int8x16ToScalar[T ~int8]()"
  - "func Int16x8ToScalar[T ~int16]()"
  - "func Int32x4ToScalar[T ~int32]()"
  - "func Int64x2ToScalar[T ~int64]()"
  - "func Uint8x16ToScalar[T ~uint8]()"
  - "func Uint16x8ToScalar[T ~uint16]()"
  - "func Uint32x4ToScalar[T ~uint32]()"
  - "func Uint64x2ToScalar[T ~uint64]()"
  - "func Float32x4ToScalar[T ~float32]()"
  - "func Float64x2ToScalar[T ~float64]()"
  - "func Int8x32ToScalar[T ~int8]()"
  - "func Int16x16ToScalar[T ~int16]()"
  - "func Int32x8ToScalar[T ~int32]()"
  - "func Int64x4ToScalar[T ~int64]()"
  - "func Uint8x32ToScalar[T ~uint8]()"
  - "func Uint16x16ToScalar[T ~uint16]()"
  - "func Uint32x8ToScalar[T ~uint32]()"
  - "func Uint64x4ToScalar[T ~uint64]()"
  - "func Float32x8ToScalar[T ~float32]()"
  - "func Float64x4ToScalar[T ~float64]()"
  - "func Int8x64ToScalar[T ~int8]()"
  - "func Int16x32ToScalar[T ~int16]()"
  - "func Int32x16ToScalar[T ~int32]()"
  - "func Int64x8ToScalar[T ~int64]()"
  - "func Uint8x64ToScalar[T ~uint8]()"
  - "func Uint16x32ToScalar[T ~uint16]()"
  - "func Uint32x16ToScalar[T ~uint32]()"
  - "func Uint64x8ToScalar[T ~uint64]()"
  - "func Float32x16ToScalar[T ~float32]()"
  - "func Float64x8ToScalar[T ~float64]()"
playUrl:
variantHelpers:
  - plugin#simd#int8x16toscalar
  - plugin#simd#int16x8toscalar
  - plugin#simd#int32x4toscalar
  - plugin#simd#int64x2toscalar
  - plugin#simd#uint8x16toscalar
  - plugin#simd#uint16x8toscalar
  - plugin#simd#uint32x4toscalar
  - plugin#simd#uint64x2toscalar
  - plugin#simd#float32x4toscalar
  - plugin#simd#float64x2toscalar
  - plugin#simd#int8x32toscalar
  - plugin#simd#int16x16toscalar
  - plugin#simd#int32x8toscalar
  - plugin#simd#int64x4toscalar
  - plugin#simd#uint8x32toscalar
  - plugin#simd#uint16x16toscalar
  - plugin#simd#uint32x8toscalar
  - plugin#simd#uint64x4toscalar
  - plugin#simd#float32x8toscalar
  - plugin#simd#float64x4toscalar
  - plugin#simd#int8x64toscalar
  - plugin#simd#int16x32toscalar
  - plugin#simd#int32x16toscalar
  - plugin#simd#int64x8toscalar
  - plugin#simd#uint8x64toscalar
  - plugin#simd#uint16x32toscalar
  - plugin#simd#uint32x16toscalar
  - plugin#simd#uint64x8toscalar
  - plugin#simd#float32x16toscalar
  - plugin#simd#float64x8toscalar
similarHelpers:
  - plugin#simd#scalartosimd
position: 10
---

Converts SIMD vectors back into streams of scalar values. Each SIMD vector emits multiple scalar values based on its lane count (2, 4, 8, 16, 32, or 64 values per vector).

```go
import (
    "fmt"

    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(1.5), float32(2.5), float32(3.5), float32(4.5),
    ),
    rosimd.ScalarToFloat32x4[float32](),
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

// Next: 1.5
// Next: 2.5
// Next: 3.5
// Next: 4.5
// Completed
```

## AVX variants (128-bit vectors)

Available on all x86_64 CPUs with AVX support (basically all modern x86_64 CPUs).

- Float32x4ToScalar
- Float64x2ToScalar
- Int8x16ToScalar
- Int16x8ToScalar
- Int32x4ToScalar
- Int64x2ToScalar
- Uint8x16ToScalar
- Uint16x8ToScalar
- Uint32x4ToScalar
- Uint64x2ToScalar


## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- Float32x8ToScalar
- Float64x4ToScalar
- Int8x32ToScalar
- Int16x16ToScalar
- Int32x8ToScalar
- Int64x4ToScalar
- Uint8x32ToScalar
- Uint16x16ToScalar
- Uint32x8ToScalar
- Uint64x4ToScalar

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- Float32x16ToScalar
- Float64x8ToScalar
- Int8x64ToScalar
- Int16x32ToScalar
- Int32x16ToScalar
- Int64x8ToScalar
- Uint8x64ToScalar
- Uint16x32ToScalar
- Uint32x16ToScalar
- Uint64x8ToScalar

