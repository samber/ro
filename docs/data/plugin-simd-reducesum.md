---
name: ReduceSum
slug: reducesum
sourceRef: plugins/exp/simd/math_sse.go
type: plugin
category: simd
signatures:
  - "func ReduceSumInt8x16[T ~int8]()"
  - "func ReduceSumInt16x8[T ~int16]()"
  - "func ReduceSumInt32x4[T ~int32]()"
  - "func ReduceSumInt64x2[T ~int64]()"
  - "func ReduceSumUint8x16[T ~uint8]()"
  - "func ReduceSumUint16x8[T ~uint16]()"
  - "func ReduceSumUint32x4[T ~uint32]()"
  - "func ReduceSumUint64x2[T ~uint64]()"
  - "func ReduceSumFloat32x4[T ~float32]()"
  - "func ReduceSumFloat64x2[T ~float64]()"
  - "func ReduceSumInt8x32[T ~int8]()"
  - "func ReduceSumInt16x16[T ~int16]()"
  - "func ReduceSumInt32x8[T ~int32]()"
  - "func ReduceSumInt64x4[T ~int64]()"
  - "func ReduceSumUint8x32[T ~uint8]()"
  - "func ReduceSumUint16x16[T ~uint16]()"
  - "func ReduceSumUint32x8[T ~uint32]()"
  - "func ReduceSumUint64x4[T ~uint64]()"
  - "func ReduceSumFloat32x8[T ~float32]()"
  - "func ReduceSumFloat64x4[T ~float64]()"
  - "func ReduceSumInt8x64[T ~int8]()"
  - "func ReduceSumInt16x32[T ~int16]()"
  - "func ReduceSumInt32x16[T ~int32]()"
  - "func ReduceSumInt64x8[T ~int64]()"
  - "func ReduceSumUint8x64[T ~uint8]()"
  - "func ReduceSumUint16x32[T ~uint16]()"
  - "func ReduceSumUint32x16[T ~uint32]()"
  - "func ReduceSumUint64x8[T ~uint64]()"
  - "func ReduceSumFloat32x16[T ~float32]()"
  - "func ReduceSumFloat64x8[T ~float64]()"
playUrl:
variantHelpers:
  - plugin#simd#reducesumint8x16
  - plugin#simd#reducesumint16x8
  - plugin#simd#reducesumint32x4
  - plugin#simd#reducesumint64x2
  - plugin#simd#reducesumuint8x16
  - plugin#simd#reducesumuint16x8
  - plugin#simd#reducesumuint32x4
  - plugin#simd#reducesumuint64x2
  - plugin#simd#reducesumfloat32x4
  - plugin#simd#reducesumfloat64x2
  - plugin#simd#reducesumint8x32
  - plugin#simd#reducesumint16x16
  - plugin#simd#reducesumint32x8
  - plugin#simd#reducesumint64x4
  - plugin#simd#reducesumuint8x32
  - plugin#simd#reducesumuint16x16
  - plugin#simd#reducesumuint32x8
  - plugin#simd#reducesumuint64x4
  - plugin#simd#reducesumfloat32x8
  - plugin#simd#reducesumfloat64x4
  - plugin#simd#reducesumint8x64
  - plugin#simd#reducesumint16x32
  - plugin#simd#reducesumint32x16
  - plugin#simd#reducesumint64x8
  - plugin#simd#reducesumuint8x64
  - plugin#simd#reducesumuint16x32
  - plugin#simd#reducesumuint32x16
  - plugin#simd#reducesumuint64x8
  - plugin#simd#reducesumfloat32x16
  - plugin#simd#reducesumfloat64x8
similarHelpers:
  - plugin#simd#reducemin
  - plugin#simd#reducemax
position: 70
---

Accumulates the sum of all lanes across SIMD vectors and emits a single scalar value when the source completes.

```go
import (
    "github.com/samber/ro"
    rosimd "github.com/samber/ro/plugins/exp/simd"
)

obs := ro.Pipe[float32, float32](
    ro.Just(
        float32(10), float32(20), float32(30), float32(40),
        float32(20), float32(40), float32(60), float32(80),
    ),
    rosimd.ScalarToFloat32x4[float32](),
    rosimd.ReduceSumFloat32x4[float32](),
)

sub := obs.Subscribe(ro.NewObserver[float32](
    func(sum float32) {
        fmt.Printf("Next: %.1f\n", sum)
    },
    ro.OnError(func(err error) {
        fmt.Printf("Error: %v\n", err)
    }),
    ro.OnComplete(func() {
        fmt.Println("Completed")
    }),
))
defer sub.Unsubscribe()

// Next: 300.0
// Completed
```

## SSE variants (128-bit vectors)

Available on all x86_64 CPUs with SSE support (basically all modern x86_64 CPUs).

- ReduceSumFloat32x4
- ReduceSumFloat64x2
- ReduceSumInt8x16
- ReduceSumInt16x8
- ReduceSumInt32x4
- ReduceSumInt64x2
- ReduceSumUint8x16
- ReduceSumUint16x8
- ReduceSumUint32x4
- ReduceSumUint64x2

## AVX2 variants (256-bit vectors)

Requires AVX2 CPU support (Intel Haswell [2013]+, AMD Ryzen [2017]+).

- ReduceSumFloat32x8
- ReduceSumFloat64x4
- ReduceSumInt8x32
- ReduceSumInt16x16
- ReduceSumInt32x8
- ReduceSumInt64x4
- ReduceSumUint8x32
- ReduceSumUint16x16
- ReduceSumUint32x8
- ReduceSumUint64x4

## AVX-512 variants (512-bit vectors)

Requires AVX-512 CPU support (Intel Skylake-X/Xeon [2017]+, AMD Zen 4 [2022]+).

- ReduceSumFloat32x16
- ReduceSumFloat64x8
- ReduceSumInt8x64
- ReduceSumInt16x32
- ReduceSumInt32x16
- ReduceSumInt64x8
- ReduceSumUint8x64
- ReduceSumUint16x32
- ReduceSumUint32x16
- ReduceSumUint64x8
