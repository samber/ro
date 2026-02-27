//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"simd/archsimd"
)

const (
	simdLanes2  = uint(2)  // Number of lanes in x64 (128-bit) SIMD vectors
	simdLanes4  = uint(4)  // Number of lanes in x128 (128-bit) SIMD vectors for 32-bit elements
	simdLanes8  = uint(8)  // Number of lanes in x256 (256-bit) SIMD vectors for 32-bit elements
	simdLanes16 = uint(16) // Number of lanes in x128 (128-bit) SIMD vectors for 8-bit elements
	simdLanes32 = uint(32) // Number of lanes in x256 (256-bit) SIMD vectors for 8-bit elements
	simdLanes64 = uint(64) // Number of lanes in x512 (512-bit) SIMD vectors for 8-bit elements
)

// simdFeature represents the highest available SIMD instruction set level
// detected at package initialization.
type simdFeature int

const (
	simdFeatureNone   simdFeature = iota // No SIMD support detected
	simdFeatureAVX                       // 128-bit AVX support
	simdFeatureAVX2                      // 256-bit AVX2 support
	simdFeatureAVX512                    // 512-bit AVX-512 support
)

// currentSimdFeature is cached at package init to avoid repeated CPU feature checks.
// This value is set once when the package is loaded based on the CPU's capabilities.
var currentSimdFeature simdFeature

func init() {
	if archsimd.X86.AVX512() {
		currentSimdFeature = simdFeatureAVX512
	} else if archsimd.X86.AVX2() {
		currentSimdFeature = simdFeatureAVX2
	} else if archsimd.X86.AVX() {
		currentSimdFeature = simdFeatureAVX
	}
}
