//go:build go1.26 && goexperiment.simd && amd64

package rosimd

import (
	"math/rand"
	"testing"
	"time"

	"github.com/samber/ro"
)

// Benchmark suite for SIMD math operations.
// These benchmarks measure the performance of Sum, Min, Max, Add, Sub, and Clamp
// operations across different SIMD implementations (AVX, AVX2, AVX512) and data sizes.

// Benchmark sizes to demonstrate performance characteristics at different scales
var benchmarkSizes = []struct {
	name string
	size int
}{
	{"small", 8},     // Smaller than typical SIMD widths
	{"medium", 128},  // Between SIMD widths
	{"large", 1024},  // Well above SIMD register widths
	{"xlarge", 8192}, // Large dataset for real-world performance
}

// Test data generators for different numeric types

func init() {
	// Seeded for reproducibility
	rand.Seed(time.Now().UnixNano())
}

func generateInt8(n int) []int8 {
	data := make([]int8, n)
	for i := range data {
		data[i] = int8(rand.Intn(127) - 64)
	}
	return data
}

func generateInt16(n int) []int16 {
	data := make([]int16, n)
	for i := range data {
		data[i] = int16(rand.Intn(32767) - 16384)
	}
	return data
}

func generateInt32(n int) []int32 {
	data := make([]int32, n)
	for i := range data {
		data[i] = int32(rand.Intn(1000) - 500)
	}
	return data
}

func generateInt64(n int) []int64 {
	data := make([]int64, n)
	for i := range data {
		data[i] = rand.Int63() % 10000
	}
	return data
}

func generateUint8(n int) []uint8 {
	data := make([]uint8, n)
	for i := range data {
		data[i] = uint8(rand.Uint32() % 256)
	}
	return data
}

func generateUint16(n int) []uint16 {
	data := make([]uint16, n)
	for i := range data {
		data[i] = uint16(rand.Uint32() % 65536)
	}
	return data
}

func generateUint32(n int) []uint32 {
	data := make([]uint32, n)
	for i := range data {
		data[i] = rand.Uint32() % 10000
	}
	return data
}

func generateUint64(n int) []uint64 {
	data := make([]uint64, n)
	for i := range data {
		data[i] = rand.Uint64() % 10000
	}
	return data
}

func generateFloat32(n int) []float32 {
	data := make([]float32, n)
	for i := range data {
		data[i] = rand.Float32()*100 - 50
	}
	return data
}

func generateFloat64(n int) []float64 {
	data := make([]float64, n)
	for i := range data {
		data[i] = rand.Float64()*100 - 50
	}
	return data
}

// ========================================
// INT8 BENCHMARKS
// ========================================

func BenchmarkReduceSumInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[int8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt8x16[int8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt8x32[int8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt8x64[int8]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMinInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[int8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt8x16[int8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt8x32[int8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt8x64[int8]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[int8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt8x16[int8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt8x32[int8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt8x64[int8]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int8) int8 { return v + 10 })(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x16ToScalar[int8]()(AddInt8x16[int8](10)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x32ToScalar[int8]()(AddInt8x32[int8](10)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x64ToScalar[int8]()(AddInt8x64[int8](10)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int8) int8 { return v - 10 })(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x16ToScalar[int8]()(SubInt8x16[int8](10)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x32ToScalar[int8]()(SubInt8x32[int8](10)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x64ToScalar[int8]()(SubInt8x64[int8](10)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampInt8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := int8(-50), int8(50)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int8) int8 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt8x16[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x16ToScalar[int8]()(ClampInt8x16[int8](-50, 50)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt8x32[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x32ToScalar[int8]()(ClampInt8x32[int8](-50, 50)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt8x64[int8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int8x64ToScalar[int8]()(ClampInt8x64[int8](-50, 50)(obs)))
				}
			})
		})
	}
}

// ========================================
// INT16 BENCHMARKS
// ========================================

func BenchmarkReduceMinInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[int16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt16x8[int16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt16x16[int16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt16x32[int16]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[int16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt16x8[int16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt16x16[int16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt16x32[int16]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[int16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt16x8[int16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt16x16[int16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt16x32[int16]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int16) int16 { return v + 100 })(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x8ToScalar[int16]()(AddInt16x8[int16](100)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x16ToScalar[int16]()(AddInt16x16[int16](100)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x32ToScalar[int16]()(AddInt16x32[int16](100)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int16) int16 { return v - 100 })(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x8ToScalar[int16]()(SubInt16x8[int16](100)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x16ToScalar[int16]()(SubInt16x16[int16](100)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x32ToScalar[int16]()(SubInt16x32[int16](100)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampInt16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := int16(-10000), int16(10000)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int16) int16 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt16x8[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x8ToScalar[int16]()(ClampInt16x8[int16](-10000, 10000)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt16x16[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x16ToScalar[int16]()(ClampInt16x16[int16](-10000, 10000)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt16x32[int16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int16x32ToScalar[int16]()(ClampInt16x32[int16](-10000, 10000)(obs)))
				}
			})
		})
	}
}

// ========================================
// INT32 BENCHMARKS
// ========================================

func BenchmarkReduceMinInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[int32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt32x4[int32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt32x8[int32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt32x16[int32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[int32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt32x4[int32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt32x8[int32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt32x16[int32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[int32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt32x4[int32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt32x8[int32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt32x16[int32]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int32) int32 { return v + 1000 })(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x4ToScalar[int32]()(AddInt32x4[int32](1000)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x8ToScalar[int32]()(AddInt32x8[int32](1000)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x16ToScalar[int32]()(AddInt32x16[int32](1000)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int32) int32 { return v - 1000 })(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x4ToScalar[int32]()(SubInt32x4[int32](1000)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x8ToScalar[int32]()(SubInt32x8[int32](1000)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x16ToScalar[int32]()(SubInt32x16[int32](1000)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampInt32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := int32(-100), int32(100)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int32) int32 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt32x4[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x4ToScalar[int32]()(ClampInt32x4[int32](-100, 100)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt32x8[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x8ToScalar[int32]()(ClampInt32x8[int32](-100, 100)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt32x16[int32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int32x16ToScalar[int32]()(ClampInt32x16[int32](-100, 100)(obs)))
				}
			})
		})
	}
}

// ========================================
// INT64 BENCHMARKS
// ========================================

func BenchmarkReduceMinInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[int64]()(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt64x2[int64]()(obs))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt64x4[int64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinInt64x8[int64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[int64]()(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt64x2[int64]()(obs))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt64x4[int64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxInt64x8[int64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[int64]()(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt64x2[int64]()(obs))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt64x4[int64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumInt64x8[int64]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int64) int64 { return v + 10000 })(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x2ToScalar[int64]()(AddInt64x2[int64](10000)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x4ToScalar[int64]()(AddInt64x4[int64](10000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x8ToScalar[int64]()(AddInt64x8[int64](10000)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int64) int64 { return v - 10000 })(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x2ToScalar[int64]()(SubInt64x2[int64](10000)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x4ToScalar[int64]()(SubInt64x4[int64](10000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x8ToScalar[int64]()(SubInt64x8[int64](10000)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampInt64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateInt64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := int64(-10000), int64(10000)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v int64) int64 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x2[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x2ToScalar[int64]()(ClampInt64x2[int64](-10000, 10000)(obs)))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x4[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x4ToScalar[int64]()(ClampInt64x4[int64](-10000, 10000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToInt64x8[int64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Int64x8ToScalar[int64]()(ClampInt64x8[int64](-10000, 10000)(obs)))
				}
			})
		})
	}
}

// ========================================
// FLOAT32 BENCHMARKS
// ========================================

func BenchmarkReduceMinFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[float32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat32x4[float32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat32x8[float32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat32x16[float32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[float32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat32x4[float32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat32x8[float32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat32x16[float32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[float32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat32x4[float32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat32x8[float32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat32x16[float32]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float32) float32 { return v + 100.0 })(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x4ToScalar[float32]()(AddFloat32x4[float32](100.0)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x8ToScalar[float32]()(AddFloat32x8[float32](100.0)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x16ToScalar[float32]()(AddFloat32x16[float32](100.0)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float32) float32 { return v - 100.0 })(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x4ToScalar[float32]()(SubFloat32x4[float32](100.0)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x8ToScalar[float32]()(SubFloat32x8[float32](100.0)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x16ToScalar[float32]()(SubFloat32x16[float32](100.0)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampFloat32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := float32(-100), float32(100)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float32) float32 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat32x4[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x4ToScalar[float32]()(ClampFloat32x4[float32](-100, 100)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat32x8[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x8ToScalar[float32]()(ClampFloat32x8[float32](-100, 100)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat32x16[float32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float32x16ToScalar[float32]()(ClampFloat32x16[float32](-100, 100)(obs)))
				}
			})
		})
	}
}

// ========================================
// FLOAT64 BENCHMARKS
// ========================================

func BenchmarkReduceMinFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[float64]()(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat64x2[float64]()(obs))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat64x4[float64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinFloat64x8[float64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[float64]()(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat64x2[float64]()(obs))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat64x4[float64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxFloat64x8[float64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[float64]()(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat64x2[float64]()(obs))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat64x4[float64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumFloat64x8[float64]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float64) float64 { return v + 100.0 })(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x2ToScalar[float64]()(AddFloat64x2[float64](100.0)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x4ToScalar[float64]()(AddFloat64x4[float64](100.0)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x8ToScalar[float64]()(AddFloat64x8[float64](100.0)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float64) float64 { return v - 100.0 })(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x2ToScalar[float64]()(SubFloat64x2[float64](100.0)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x4ToScalar[float64]()(SubFloat64x4[float64](100.0)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x8ToScalar[float64]()(SubFloat64x8[float64](100.0)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampFloat64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateFloat64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := float64(-100), float64(100)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v float64) float64 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToFloat64x2[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x2ToScalar[float64]()(ClampFloat64x2[float64](-100, 100)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToFloat64x4[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x4ToScalar[float64]()(ClampFloat64x4[float64](-100, 100)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToFloat64x8[float64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Float64x8ToScalar[float64]()(ClampFloat64x8[float64](-100, 100)(obs)))
				}
			})
		})
	}
}

// ========================================
// UNSIGNED INTEGER BENCHMARKS
// ========================================

func BenchmarkReduceMinUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[uint8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint8x16[uint8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint8x32[uint8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint8x64[uint8]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[uint8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint8x16[uint8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint8x32[uint8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint8x64[uint8]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[uint8]()(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint8x16[uint8]()(obs))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint8x32[uint8]()(obs))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint8x64[uint8]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint8) uint8 { return v + 10 })(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x16ToScalar[uint8]()(AddUint8x16[uint8](10)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x32ToScalar[uint8]()(AddUint8x32[uint8](10)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x64ToScalar[uint8]()(AddUint8x64[uint8](10)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint8) uint8 {
						if v < 10 {
							return 0
						}
						return v - 10
					})(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x16ToScalar[uint8]()(SubUint8x16[uint8](10)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x32ToScalar[uint8]()(SubUint8x32[uint8](10)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x64ToScalar[uint8]()(SubUint8x64[uint8](10)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampUint8(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint8(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := uint8(50), uint8(200)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint8) uint8 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x16", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint8x16[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x16ToScalar[uint8]()(ClampUint8x16[uint8](50, 200)(obs)))
				}
			})

			b.Run("AVX2-x32", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint8x32[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x32ToScalar[uint8]()(ClampUint8x32[uint8](50, 200)(obs)))
				}
			})

			b.Run("AVX512-x64", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint8x64[uint8]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint8x64ToScalar[uint8]()(ClampUint8x64[uint8](50, 200)(obs)))
				}
			})
		})
	}
}

func BenchmarkReduceMinUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[uint16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint16x8[uint16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint16x16[uint16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint16x32[uint16]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[uint16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint16x8[uint16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint16x16[uint16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint16x32[uint16]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[uint16]()(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint16x8[uint16]()(obs))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint16x16[uint16]()(obs))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint16x32[uint16]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint16) uint16 { return v + 100 })(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x8ToScalar[uint16]()(AddUint16x8[uint16](100)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x16ToScalar[uint16]()(AddUint16x16[uint16](100)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x32ToScalar[uint16]()(AddUint16x32[uint16](100)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint16) uint16 {
						if v < 100 {
							return 0
						}
						return v - 100
					})(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x8ToScalar[uint16]()(SubUint16x8[uint16](100)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x16ToScalar[uint16]()(SubUint16x16[uint16](100)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x32ToScalar[uint16]()(SubUint16x32[uint16](100)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampUint16(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint16(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := uint16(5000), uint16(60000)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint16) uint16 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x8", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint16x8[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x8ToScalar[uint16]()(ClampUint16x8[uint16](5000, 60000)(obs)))
				}
			})

			b.Run("AVX2-x16", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint16x16[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x16ToScalar[uint16]()(ClampUint16x16[uint16](5000, 60000)(obs)))
				}
			})

			b.Run("AVX512-x32", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint16x32[uint16]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint16x32ToScalar[uint16]()(ClampUint16x32[uint16](5000, 60000)(obs)))
				}
			})
		})
	}
}

func BenchmarkReduceMinUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[uint32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint32x4[uint32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint32x8[uint32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint32x16[uint32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[uint32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint32x4[uint32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint32x8[uint32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint32x16[uint32]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[uint32]()(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint32x4[uint32]()(obs))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint32x8[uint32]()(obs))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint32x16[uint32]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint32) uint32 { return v + 1000 })(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x4ToScalar[uint32]()(AddUint32x4[uint32](1000)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x8ToScalar[uint32]()(AddUint32x8[uint32](1000)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x16ToScalar[uint32]()(AddUint32x16[uint32](1000)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint32) uint32 {
						if v < 1000 {
							return 0
						}
						return v - 1000
					})(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x4ToScalar[uint32]()(SubUint32x4[uint32](1000)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x8ToScalar[uint32]()(SubUint32x8[uint32](1000)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x16ToScalar[uint32]()(SubUint32x16[uint32](1000)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampUint32(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint32(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := uint32(100000), uint32(900000)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint32) uint32 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX-x4", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint32x4[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x4ToScalar[uint32]()(ClampUint32x4[uint32](100000, 900000)(obs)))
				}
			})

			b.Run("AVX2-x8", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint32x8[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x8ToScalar[uint32]()(ClampUint32x8[uint32](100000, 900000)(obs)))
				}
			})

			b.Run("AVX512-x16", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint32x16[uint32]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint32x16ToScalar[uint32]()(ClampUint32x16[uint32](100000, 900000)(obs)))
				}
			})
		})
	}
}

func BenchmarkReduceMinUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Min[uint64]()(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint64x2[uint64]()(obs))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint64x4[uint64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMinUint64x8[uint64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceMaxUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Max[uint64]()(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint64x2[uint64]()(obs))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint64x4[uint64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceMaxUint64x8[uint64]()(obs))
				}
			})
		})
	}
}

func BenchmarkReduceSumUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Sum[uint64]()(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint64x2[uint64]()(obs))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint64x4[uint64]()(obs))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ReduceSumUint64x8[uint64]()(obs))
				}
			})
		})
	}
}

func BenchmarkAddUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint64) uint64 { return v + 10000 })(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x2ToScalar[uint64]()(AddUint64x2[uint64](10000)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x4ToScalar[uint64]()(AddUint64x4[uint64](10000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x8ToScalar[uint64]()(AddUint64x8[uint64](10000)(obs)))
				}
			})
		})
	}
}

func BenchmarkSubUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint64) uint64 {
						if v < 10000 {
							return 0
						}
						return v - 10000
					})(obs))
				}
			})

			b.Run("AVX-x2", func(b *testing.B) {
				requireAVX(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x2ToScalar[uint64]()(SubUint64x2[uint64](10000)(obs)))
				}
			})

			b.Run("AVX2-x4", func(b *testing.B) {
				requireAVX2(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x4ToScalar[uint64]()(SubUint64x4[uint64](10000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x8ToScalar[uint64]()(SubUint64x8[uint64](10000)(obs)))
				}
			})
		})
	}
}

func BenchmarkClampUint64(b *testing.B) {
	for _, bs := range benchmarkSizes {
		data := generateUint64(bs.size)

		b.Run(bs.name, func(b *testing.B) {
			b.Run("fallback-ro", func(b *testing.B) {
				minVal, maxVal := uint64(100000000), uint64(900000000)
				obs := ro.Just(data...)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(ro.Map(func(v uint64) uint64 {
						if v < minVal {
							return minVal
						}
						if v > maxVal {
							return maxVal
						}
						return v
					})(obs))
				}
			})

			b.Run("AVX512-x2", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x2[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x2ToScalar[uint64]()(ClampUint64x2[uint64](100000000, 900000000)(obs)))
				}
			})

			b.Run("AVX512-x4", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x4[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x4ToScalar[uint64]()(ClampUint64x4[uint64](100000000, 900000000)(obs)))
				}
			})

			b.Run("AVX512-x8", func(b *testing.B) {
				requireAVX512(b)
				obs := ScalarToUint64x8[uint64]()(ro.Just(data...))
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					_, _ = ro.Collect(Uint64x8ToScalar[uint64]()(ClampUint64x8[uint64](100000000, 900000000)(obs)))
				}
			})
		})
	}
}
