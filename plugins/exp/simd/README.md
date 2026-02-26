# exp/simd - SIMD-Accelerated Operators for ro

This package provides SIMD-accelerated mathematical operators for the [ro](../../) reactive observables library, leveraging Go's experimental SIMD support for high-performance data processing on AMD64 processors.

## Requirements

- Go 1.26 or later
- AMD64 architecture
- `GOEXPERIMENT=simd` environment variable must be set

```bash
export GOEXPERIMENT=simd
```

## Architecture

The package automatically detects available CPU features at runtime and dispatches to the most efficient implementation:

| Instruction Set | Vector Width | Lanes (int8) | Lanes (float32) | Detection               |
| --------------- | ------------ | ------------ | --------------- | ----------------------- |
| None (fallback) | N/A          | 1            | 1               | Default                 |
| SSE (AVX)       | 128-bit      | 16           | 4               | `archsimd.X86.AVX()`    |
| AVX2            | 256-bit      | 32           | 8               | `archsimd.X86.AVX2()`   |
| AVX-512         | 512-bit      | 64           | 16              | `archsimd.X86.AVX512()` |

CPU feature detection is performed once at package initialization for maximum performance.

## Supported Types

All integer and floating-point types are supported:

- **Signed integers**: `int8`, `int16`, `int32`, `int64`
- **Unsigned integers**: `uint8`, `uint16`, `uint32`, `uint64`
- **Floating-point**: `float32`, `float64`

## API

### Working with SIMD Vectors

This library operates on SIMD vector types (e.g., `Int32x4`, `Float32x4`) rather than scalar values. To process scalar data:

1. **Convert scalars to SIMD vectors** using `ScalarTo[Type]x[N]`
2. **Apply operations** to the vectors
3. **Convert back to scalars** using `[Type]x[N]ToScalar`

### Arithmetic Operators

Add or subtract a constant value from each element:

```go
// Add 10 to each int32 value
result := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8),
    simd.ScalarToInt32x4[int32](),
    simd.AddInt32x4[int32](10),
    simd.Int32x4ToScalar[int32](),
).Collect() // [11, 12, 13, 14, 15, 16, 17, 18]

// Subtract 5 from each float32 value
result := ro.Pipe(
    ro.Just(1.5, 2.5, 3.5, 4.5),
    simd.ScalarToFloat32x4[float32](),
    simd.SubFloat32x4[float32](5.0),
    simd.Float32x4ToScalar[float32](),
).Collect() // [-3.5, -2.5, -1.5, -0.5]
```

### Comparison Operators

Clamp values to a range:

```go
// Clamp int8 values between 0 and 100
result := ro.Pipe(
    ro.Just(-5, 50, 150, -10, 200),
    simd.ScalarToInt8x16[int8](),
    simd.ClampInt8x16[int8](0, 100),
    simd.Int8x16ToScalar[int8](),
).Collect() // [0, 50, 100, 0, 100, ...]
```

Apply minimum/maximum constraints:

```go
// Ensure no value is below -10
result := ro.Pipe(
    ro.Just(-20, -5, 10, -30),
    simd.ScalarToInt32x4[int32](),
    simd.MinInt32x4[int32](-10),
    simd.Int32x4ToScalar[int32](),
).Collect() // [-10, -5, 10, -10]

// Ensure no value is above 100
result := ro.Pipe(
    ro.Just(50, 100, 150, 200),
    simd.ScalarToInt32x4[int32](),
    simd.MaxInt32x4[int32](100),
    simd.Int32x4ToScalar[int32](),
).Collect() // [50, 100, 100, 100]
```

### Reduction Operators

Compute aggregates efficiently:

```go
// Sum all int32 values
sum := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8),
    simd.ScalarToInt32x4[int32](),
    simd.ReduceSumInt32x4[int32](),
).Collect() // 36

// Find minimum float64 value
min := ro.Pipe(
    ro.Just(1.5, 0.5, 2.5, 3.0),
    simd.ScalarToFloat64x2[float64](),
    simd.ReduceMinFloat64x2[float64](),
).Collect() // 0.5

// Find maximum int8 value
max := ro.Pipe(
    ro.Just(10, 20, 15, 5, 25, 30, 12, 18, 8, 22, 14, 16, 3, 28, 7, 19),
    simd.ScalarToInt8x16[int8](),
    simd.ReduceMaxInt8x16[int8](),
).Collect() // 30
```

### Available Operators

Operators are available for all numeric types with vector width suffixes:

| Type    | Vectors   | Arithmetic | Comparison      | Reduction                       |
| ------- | --------- | ---------- | --------------- | ------------------------------- |
| int8    | Int8x16   | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| int16   | Int16x8   | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| int32   | Int32x4   | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| int64   | Int64x2   | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| uint8   | Uint8x16  | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| uint16  | Uint16x8  | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| uint32  | Uint32x4  | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| uint64  | Uint64x2  | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| float32 | Float32x4 | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |
| float64 | Float64x2 | Add, Sub   | Clamp, Min, Max | ReduceSum, ReduceMin, ReduceMax |

## Performance Characteristics

SIMD operations provide significant speedup for:

- **Batch operations**: Processing many elements at once
- **Large datasets**: Data larger than cache lines benefits most
- **Parallel-friendly patterns**: Element-wise operations

Performance improvements scale with:
1. **Vector width**: AVX-512 (512-bit) > AVX2 (256-bit) > SSE (128-bit)
2. **Element size**: `int8` (64 lanes) > `float32` (16 lanes) > `float64` (8 lanes)

### Example Benchmarks

Typical speedup on AVX-512 systems:

| Operation | Type    | Speedup vs Baseline |
| --------- | ------- | ------------------- |
| Add       | int8    | ~50-60x             |
| Add       | float32 | ~12-15x             |
| ReduceSum | int8    | ~40-50x             |
| ReduceSum | float32 | ~10-12x             |

*Actual performance varies by CPU model, data size, and memory access patterns.*

## Implementation Notes

### Scalar Broadcasting for Add/Sub

Arithmetic operators (`Add`, `Sub`) now use efficient scalar broadcasting internally. When adding or subtracting a scalar value, the value is broadcast across all lanes of the SIMD vector:

```go
// Example: AddInt8x16 implementation
vector := archsimd.BroadcastInt8x16(int8(number))
added := value.Add(vector)
```

This approach provides:
- **Cleaner API**: You pass scalar values directly
- **Optimal performance**: Single broadcast instruction before vectorized operation
- **Consistent semantics**: Same interface as non-SIMD fallback

### Conversion Operators

The package includes `ScalarTo[Type]x[N]` and `[Type]x[N]ToScalar` operators for converting between scalar streams and SIMD vectors:

```go
// Convert scalar stream to Int8x16 vectors
vectors := ro.Pipe(
    ro.Just(1, 2, ..., 16, 17, 18, ...),
    simd.ScalarToInt8x16[int8](),
)

// Convert Int8x16 vectors back to scalars
scalars := ro.Pipe(
    vectors,
    simd.Int8x16ToScalar[int8](),
)
```

### Buffer-Based Reductions

Reduce operations use a buffer-based approach for maximum efficiency:

```go
var buf [lanes]int32
accumulation.Store(&buf)
total := int32(0)
for i := uint(0); i < lanes; i++ {
    total += buf[i]
}
```

This avoids the overhead of element-wise `GetElem` calls.

### Fallback Behavior

On systems without SIMD support or non-AMD64 architectures, all operators fall back to equivalent `ro.Map` and `ro.Reduce` implementations, ensuring correctness everywhere while maximizing performance on supported hardware.

## Testing

Run tests with SIMD experiment enabled:

```bash
GOEXPERIMENT=simd go test ./plugins/exp/simd/...
```

Run benchmarks:

```bash
GOEXPERIMENT=simd go test -bench=. ./plugins/exp/simd/...
```

### Test Files

- `simd_test.go` - Core operator tests
- `math_sse_test.go` - SSE-specific math tests
- `math_avx2_test.go` - AVX2-specific math tests
- `math_avx512_test.go` - AVX-512-specific math tests
- `conversion_sse_test.go` - SSE conversion operator tests
- `conversion_avx2_test.go` - AVX2 conversion operator tests
- `conversion_avx512_test.go` - AVX-512 conversion operator tests
- `math_bench_test.go` - Performance benchmarks
- `cpu_amd64_test.go` - CPU feature detection tests

## Building

Build your application with SIMD support:

```bash
GOEXPERIMENT=simd go build ./...
```

For Windows:
```powershell
$env:GOEXPERIMENT="simd"; go build ./...
```

## File Organization

```
plugins/exp/simd/
├── README.md                    # This file
├── go.mod                       # Module definition with SIMD dependency
├── simd.go                      # Fallback for non-amd64 systems
├── cpu_amd64.go                 # CPU feature detection
├── math_sse.go                  # SSE implementations (128-bit)
├── math_avx2.go                 # AVX2 implementations (256-bit)
├── math_avx512.go               # AVX-512 implementations (512-bit)
├── conversion_sse.go            # SSE conversion operators
├── conversion_avx2.go           # AVX2 conversion operators
├── conversion_avx512.go         # AVX-512 conversion operators
├── *test.go                     # Test and benchmark files
└── *.go                         # Additional utilities
```

## Contributing

When adding new operators:

1. Implement in `math_avx.go`, `math_avx2.go`, and `math_avx512.go`
2. Add tests in each architecture-specific test file (`math_avx_test.go`, `math_avx2_test.go`, `math_avx512_test.go`)
3. Add benchmarks in `math_bench_test.go`
4. Ensure fallback behavior works correctly (non-AMD64 platforms)
5. Add documentation in /docs/data and /docs/static/llms.txt

## License

Same as parent [ro](../../) project.
