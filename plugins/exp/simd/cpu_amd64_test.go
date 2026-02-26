//go:build go1.26 && goexperiment.simd && amd64

package simd

import (
	"fmt"
	"os"
	"testing"

	"simd/archsimd"

	"github.com/stretchr/testify/assert"
)

// skipHelper is a small interface implemented by both *testing.T and *testing.B
// to allow unified CPU feature requirement checking for both tests and benchmarks.
type skipHelper interface {
	Helper()
	Skipf(format string, args ...any)
}

// How to check if your Linux CPU supports SIMD (avoids SIGILL):
//
//   grep -E 'avx|sse' /proc/cpuinfo
//
// Or:  lscpu | grep -i avx
//
// You need:
//   - SSE tests (128-bit):  sse2 (baseline on amd64), sse4.1/sse4.2 often used
//   - AVX2 tests (256-bit):  avx2  in flags
//   - AVX-512 tests:        avx512f (and often avx512bw, avx512vl)
//
// If your CPU lacks AVX2 or AVX-512, tests that use them will be skipped automatically.

// requireAVX2 skips the test/benchmark if the CPU does not support AVX2 (256-bit SIMD).
// Use at the start of each AVX2 test/benchmark to avoid SIGILL on older or non-x86 systems.
func requireAVX2(t skipHelper) {
	t.Helper()
	if !archsimd.X86.AVX2() {
		t.Skipf("CPU does not support AVX2; skipping. Check compatibility: grep avx2 /proc/cpuinfo")
	}
}

// requireAVX512 skips the test/benchmark if the CPU does not support AVX-512 Foundation.
// Use at the start of each AVX-512 test/benchmark to avoid SIGILL on CPUs without AVX-512.
func requireAVX512(t skipHelper) {
	t.Helper()
	if !archsimd.X86.AVX512() {
		t.Skipf("CPU does not support AVX-512; skipping. Check compatibility: grep avx512 /proc/cpuinfo")
	}
}

// PrintCPUFeatures prints detected x86 SIMD features (for debugging).
// Run: go test -run PrintCPUFeatures -v
func PrintCPUFeatures(t *testing.T) {
	fmt.Fprintf(os.Stdout, "X86 HasAVX=%v HasAVX2=%v HasAVX512=%v\n",
		archsimd.X86.AVX(), archsimd.X86.AVX2(), archsimd.X86.AVX512())
}

// ==================== Test Helpers ====================

// verifyInt8VectorContent checks that a vector contains the expected values
func verifyInt8VectorContent(t *testing.T, vec *archsimd.Int8x16, expected []int8) {
	t.Helper()
	var buf [16]int8
	vec.Store(&buf)
	for i, exp := range expected {
		if buf[i] != exp {
			t.Errorf("element %d: expected %d, got %d", i, exp, buf[i])
		}
	}
}

// verifyFloat32VectorContent checks that a float32 vector contains expected values.
// Uses delta comparison to avoid floating-point precision issues.
func verifyFloat32VectorContent(t *testing.T, vec *archsimd.Float32x4, expected []float32) {
	t.Helper()
	is := assert.New(t)
	var buf [4]float32
	vec.Store(&buf)
	for i, exp := range expected {
		is.InDelta(exp, buf[i], 0.0001, "element %d", i)
	}
}

// ==================== Helper Functions ====================

func makeInt8Range(start, end int) []int8 {
	v := make([]int8, end-start+1)
	for i := range v {
		v[i] = int8(start + i)
	}
	return v
}

func makeInt8Filled(n int, val int8) []int8 {
	v := make([]int8, n)
	for i := range v {
		v[i] = val
	}
	return v
}

func makeInt16Range(start, end int) []int16 {
	v := make([]int16, end-start+1)
	for i := range v {
		v[i] = int16(start + i)
	}
	return v
}

func makeInt32Range(start, end int) []int32 {
	v := make([]int32, end-start+1)
	for i := range v {
		v[i] = int32(start + i)
	}
	return v
}

func makeInt64Range(start, end int) []int64 {
	v := make([]int64, end-start+1)
	for i := range v {
		v[i] = int64(start + i)
	}
	return v
}

func makeFloat32Range(start, end int) []float32 {
	v := make([]float32, end-start+1)
	for i := range v {
		v[i] = float32(start + i)
	}
	return v
}

func makeUint8Filled(n int, val uint8) []uint8 {
	v := make([]uint8, n)
	for i := range v {
		v[i] = val
	}
	return v
}

func makeUint16Filled(n int, val uint16) []uint16 {
	v := make([]uint16, n)
	for i := range v {
		v[i] = val
	}
	return v
}

func makeUint32Filled(n int, val uint32) []uint32 {
	v := make([]uint32, n)
	for i := range v {
		v[i] = val
	}
	return v
}

func makeUint64Range(start, end int) []uint64 {
	v := make([]uint64, end-start+1)
	for i := range v {
		v[i] = uint64(start + i)
	}
	return v
}
