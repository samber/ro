// Package rosimd provides SIMD-accelerated mathematical operators for the ro reactive
// observables library. It leverages Go's experimental SIMD support for high-performance
// data processing on AMD64 processors.
//
// # Requirements
//
//   - Go 1.26 or later
//   - AMD64 architecture
//   - GOEXPERIMENT=simd environment variable must be set
//
// # Architecture Support
//
// The package automatically detects available CPU features at package initialization
// and dispatches to the most efficient implementation:
//   - AVX: 128-bit vectors, 16 int8 lanes, 4 float32 lanes
//   - AVX2: 256-bit vectors, 32 int8 lanes, 8 float32 lanes
//   - AVX-512: 512-bit vectors, 64 int8 lanes, 16 float32 lanes
//
// # Overflow Behavior
//
// Integer reduction operations (ReduceSum, ReduceMin, ReduceMax) use standard
// CPU arithmetic which wraps around on overflow (two's complement semantics).
// This is intentional for SIMD performance but differs from Go's default
// panic behavior for overflow in some cases. For example, summing 1 through
// 16 using ReduceSumInt8x16 will wrap to -120 because 136 - 256 = -120.
//
// # Type Parameter Usage
//
// All operators use Go type parameters with constraints like ~int8, ~float32, etc.
// This allows them to work with type aliases (e.g., type MyInt32 int32).
//
// # Example Usage
//
//	// Add 10 to each int32 value
//	result := ro.Pipe(
//	    ro.Just(1, 2, 3, 4, 5, 6, 7, 8),
//	    rosimd.ScalarToInt32x4[int32](),
//	    rosimd.AddInt32x4[int32](10),
//	    rosimd.Int32x4ToScalar[int32](),
//	)
//
// # Fallback Behavior
//
// On non-AMD64 architectures or systems without SIMD support, all operators fall
// back to equivalent ro.Map and ro.Reduce implementations, ensuring correctness
// everywhere while maximizing performance on supported hardware.

package rosimd

// Empty file to satisfy the build constraint for non-supported architectures.
