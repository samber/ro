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
	"fmt"
	"testing"
)

func TestStringHashers_FNV64a(t *testing.T) {
	hasher := StringHash.FNV64a()

	t.Run("same input", func(t *testing.T) {
		result1 := hasher("test")
		result2 := hasher("test")

		if result1 != result2 {
			t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
		}
	})

	t.Run("different input", func(t *testing.T) {
		result1 := hasher("test")
		result3 := hasher("different")

		if result1 == result3 {
			t.Errorf("Different input should produce different hash, got %d for both", result1)
		}
	})

	t.Run("empty string", func(t *testing.T) {
		emptyHash := hasher("")
		if emptyHash == 0 {
			t.Error("Empty string should not produce zero hash")
		}
	})

	t.Run("long string", func(t *testing.T) {
		longString := "this is a very long string that should be hashed properly"
		longHash := hasher(longString)
		if longHash == 0 {
			t.Error("Long string should not produce zero hash")
		}
	})
}

func TestStringHashers_FNV64(t *testing.T) {
	hasher := StringHash.FNV64()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_FNV32a(t *testing.T) {
	hasher := StringHash.FNV32a()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_FNV32(t *testing.T) {
	hasher := StringHash.FNV32()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_SHA256(t *testing.T) {
	hasher := StringHash.SHA256()

	t.Run("same input", func(t *testing.T) {
		result1 := hasher("test")
		result2 := hasher("test")

		if result1 != result2 {
			t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
		}
	})

	t.Run("different input", func(t *testing.T) {
		result1 := hasher("test")
		result3 := hasher("different")

		if result1 == result3 {
			t.Errorf("Different input should produce different hash, got %d for both", result1)
		}
	})

	t.Run("differs from FNV", func(t *testing.T) {
		fnvHasher := StringHash.FNV64a()
		fnvResult := fnvHasher("test")
		if hasher("test") == fnvResult {
			t.Error("SHA256 should produce different results than FNV")
		}
	})
}

func TestStringHashers_SHA1(t *testing.T) {
	hasher := StringHash.SHA1()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_SHA512(t *testing.T) {
	hasher := StringHash.SHA512()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_MD5(t *testing.T) {
	hasher := StringHash.MD5()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_MapHash(t *testing.T) {
	hasher := StringHash.MapHash()

	// MapHash is non-deterministic, so we only test that it produces non-zero results
	result1 := hasher("test")
	result2 := hasher("different")

	if result1 == 0 {
		t.Error("MapHash should not produce zero hash")
	}

	if result2 == 0 {
		t.Error("MapHash should not produce zero hash")
	}
}

func TestStringHashers_CRC32(t *testing.T) {
	hasher := StringHash.CRC32()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_CRC64(t *testing.T) {
	hasher := StringHash.CRC64()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_Adler32(t *testing.T) {
	hasher := StringHash.Adler32()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_Jenkins(t *testing.T) {
	hasher := StringHash.Jenkins()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_DJB2(t *testing.T) {
	hasher := StringHash.DJB2()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_SDBM(t *testing.T) {
	hasher := StringHash.SDBM()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringHashers_Loselose(t *testing.T) {
	hasher := StringHash.Loselose()

	result1 := hasher("test")
	result2 := hasher("test")
	result3 := hasher("different")

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_FNV64a(t *testing.T) {
	hasher := BytesHash.FNV64a()

	t.Run("same input", func(t *testing.T) {
		result1 := hasher([]byte("test"))
		result2 := hasher([]byte("test"))

		if result1 != result2 {
			t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
		}
	})

	t.Run("different input", func(t *testing.T) {
		result1 := hasher([]byte("test"))
		result3 := hasher([]byte("different"))

		if result1 == result3 {
			t.Errorf("Different input should produce different hash, got %d for both", result1)
		}
	})

	t.Run("empty byte slice", func(t *testing.T) {
		emptyHash := hasher([]byte{})
		if emptyHash == 0 {
			t.Error("Empty byte slice should not produce zero hash")
		}
	})
}

func TestBytesHashers_FNV64(t *testing.T) {
	hasher := BytesHash.FNV64()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_FNV32a(t *testing.T) {
	hasher := BytesHash.FNV32a()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_FNV32(t *testing.T) {
	hasher := BytesHash.FNV32()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_SHA256(t *testing.T) {
	hasher := BytesHash.SHA256()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_SHA1(t *testing.T) {
	hasher := BytesHash.SHA1()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_SHA512(t *testing.T) {
	hasher := BytesHash.SHA512()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_MD5(t *testing.T) {
	hasher := BytesHash.MD5()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_MapHash(t *testing.T) {
	hasher := BytesHash.MapHash()

	// MapHash is non-deterministic, so we only test that it produces non-zero results
	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("different"))

	if result1 == 0 {
		t.Error("MapHash should not produce zero hash")
	}

	if result2 == 0 {
		t.Error("MapHash should not produce zero hash")
	}
}

func TestBytesHashers_CRC32(t *testing.T) {
	hasher := BytesHash.CRC32()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_CRC64(t *testing.T) {
	hasher := BytesHash.CRC64()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_Adler32(t *testing.T) {
	hasher := BytesHash.Adler32()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_Jenkins(t *testing.T) {
	hasher := BytesHash.Jenkins()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_DJB2(t *testing.T) {
	hasher := BytesHash.DJB2()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_SDBM(t *testing.T) {
	hasher := BytesHash.SDBM()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestBytesHashers_Loselose(t *testing.T) {
	hasher := BytesHash.Loselose()

	result1 := hasher([]byte("test"))
	result2 := hasher([]byte("test"))
	result3 := hasher([]byte("different"))

	if result1 != result2 {
		t.Errorf("Same input should produce same hash, got %d and %d", result1, result2)
	}

	if result1 == result3 {
		t.Errorf("Different input should produce different hash, got %d for both", result1)
	}
}

func TestStringVsBytesConsistency(t *testing.T) {
	// Test that string and bytes hashers produce consistent results
	testData := "test string"

	stringFNV := StringHash.FNV64a()(testData)
	bytesFNV := BytesHash.FNV64a()([]byte(testData))

	if stringFNV != bytesFNV {
		t.Errorf("String and bytes FNV should produce same result, got %d and %d", stringFNV, bytesFNV)
	}

	stringSHA := StringHash.SHA256()(testData)
	bytesSHA := BytesHash.SHA256()([]byte(testData))

	if stringSHA != bytesSHA {
		t.Errorf("String and bytes SHA256 should produce same result, got %d and %d", stringSHA, bytesSHA)
	}
}

func TestHashDistribution(t *testing.T) {
	// Test that hashes are reasonably distributed
	hasher := StringHash.FNV64a()
	results := make(map[uint64]bool)

	// Generate hashes for many different inputs
	for i := 0; i < 1000; i++ {
		input := fmt.Sprintf("test%d", i)
		hash := hasher(input)
		results[hash] = true
	}

	// Check that we have a reasonable number of unique hashes
	// (should be close to 1000, but allow for some collisions)
	if len(results) < 900 {
		t.Errorf("Hash distribution seems poor, got %d unique hashes out of 1000", len(results))
	}
}

func TestHashCollisionResistance(t *testing.T) {
	// Test that similar inputs produce different hashes
	hasher := StringHash.FNV64a()

	variations := []string{
		"test",
		"Test",
		"TEST",
		"test ",
		" test",
		"test1",
		"1test",
		"test\n",
		"test\t",
	}

	results := make(map[uint64]bool)
	for _, variation := range variations {
		hash := hasher(variation)
		if results[hash] {
			t.Errorf("Hash collision detected for variations of 'test': %s", variation)
		}
		results[hash] = true
	}
}
