// Copyright 2025 samber.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.apache.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rohyperloglog

import (
	cryptorand "crypto/rand"
	"fmt"
	mathrand "math/rand"
	"testing"
	"time"
)

var rng = mathrand.New(mathrand.NewSource(time.Now().UnixNano()))

// generateTestData generates test data for benchmarking
func generateTestData(count int) ([]string, [][]byte) {
	strings := make([]string, count)
	bytes := make([][]byte, count)

	for i := 0; i < count; i++ {
		// Generate random string
		randBytes := make([]byte, 16)
		cryptorand.Read(randBytes)
		strings[i] = fmt.Sprintf("test_%d_%x", i, randBytes)

		// Generate random bytes
		bytes[i] = make([]byte, 16)
		cryptorand.Read(bytes[i])
	}

	return strings, bytes
}

// BenchmarkStringHashers benchmarks all string hashing algorithms
func BenchmarkStringHashers(b *testing.B) {
	data, _ := generateTestData(1000)

	hashers := map[string]func(string) uint64{
		"FNV64a":   StringHash.FNV64a(),
		"FNV64":    StringHash.FNV64(),
		"FNV32a":   StringHash.FNV32a(),
		"FNV32":    StringHash.FNV32(),
		"SHA256":   StringHash.SHA256(),
		"SHA1":     StringHash.SHA1(),
		"SHA512":   StringHash.SHA512(),
		"MD5":      StringHash.MD5(),
		"MapHash":  StringHash.MapHash(),
		"CRC32":    StringHash.CRC32(),
		"CRC64":    StringHash.CRC64(),
		"Adler32":  StringHash.Adler32(),
		"Jenkins":  StringHash.Jenkins(),
		"DJB2":     StringHash.DJB2(),
		"SDBM":     StringHash.SDBM(),
		"Loselose": StringHash.Loselose(),
	}

	for name, hasher := range hashers {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for _, item := range data {
					hasher(item)
				}
			}
		})
	}
}

// BenchmarkBytesHashers benchmarks all bytes hashing algorithms
func BenchmarkBytesHashers(b *testing.B) {
	_, data := generateTestData(1000)

	hashers := map[string]func([]byte) uint64{
		"FNV64a":   BytesHash.FNV64a(),
		"FNV64":    BytesHash.FNV64(),
		"FNV32a":   BytesHash.FNV32a(),
		"FNV32":    BytesHash.FNV32(),
		"SHA256":   BytesHash.SHA256(),
		"SHA1":     BytesHash.SHA1(),
		"SHA512":   BytesHash.SHA512(),
		"MD5":      BytesHash.MD5(),
		"MapHash":  BytesHash.MapHash(),
		"CRC32":    BytesHash.CRC32(),
		"CRC64":    BytesHash.CRC64(),
		"Adler32":  BytesHash.Adler32(),
		"Jenkins":  BytesHash.Jenkins(),
		"DJB2":     BytesHash.DJB2(),
		"SDBM":     BytesHash.SDBM(),
		"Loselose": BytesHash.Loselose(),
	}

	for name, hasher := range hashers {
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				for _, item := range data {
					hasher(item)
				}
			}
		})
	}
}
