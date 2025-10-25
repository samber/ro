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

package robytes

import (
	"bytes"
	"math"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xrand"
)

var (
	LowerCaseLettersCharset = []rune("abcdefghijklmnopqrstuvwxyz")
	UpperCaseLettersCharset = []rune("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
	LettersCharset          = append(LowerCaseLettersCharset, UpperCaseLettersCharset...)
	NumbersCharset          = []rune("0123456789")
	AlphanumericCharset     = append(LettersCharset, NumbersCharset...)
	SpecialCharset          = []rune("!@#$%^&*()_+-=[]{}|;':\",./<>?")
	AllCharset              = append(AlphanumericCharset, SpecialCharset...)

	maximumCapacity = math.MaxInt>>1 + 1
)

// nearestPowerOfTwo returns the nearest power of two.
func nearestPowerOfTwo(cap int) int {
	n := cap - 1
	n |= n >> 1
	n |= n >> 2
	n |= n >> 4
	n |= n >> 8
	n |= n >> 16
	if n < 0 {
		return 1
	}
	if n >= maximumCapacity {
		return maximumCapacity
	}
	return n + 1
}

func random(size int, charset []rune) []byte {
	// see https://stackoverflow.com/questions/22892120/how-to-generate-a-random-string-of-a-fixed-length-in-go
	sb := bytes.Buffer{}
	sb.Grow(size)
	// Calculate the number of bits required to represent the charset,
	// e.g., for 62 characters, it would need 6 bits (since 62 -> 64 = 2^6)
	letterIdBits := int(math.Log2(float64(nearestPowerOfTwo(len(charset)))))
	// Determine the corresponding bitmask,
	// e.g., for 62 characters, the bitmask would be 111111.
	var letterIdMask int64 = 1<<letterIdBits - 1
	// Available count, since xrand.Int64() returns a non-negative number, the first bit is fixed, so there are 63 random bits
	// e.g., for 62 characters, this value is 10 (63 / 6).
	letterIdMax := 63 / letterIdBits
	// Generate the random string in a loop.
	for i, cache, remain := size-1, xrand.Int64(), letterIdMax; i >= 0; {
		// Regenerate the random number if all available bits have been used
		if remain == 0 {
			cache, remain = xrand.Int64(), letterIdMax
		}
		// Select a character from the charset
		if idx := int(cache & letterIdMask); idx < len(charset) {
			sb.WriteRune(charset[idx])
			i--
		}
		// Shift the bits to the right to prepare for the next character selection,
		// e.g., for 62 characters, shift by 6 bits.
		cache >>= letterIdBits
		// Decrease the remaining number of uses for the current random number.
		remain--
	}
	return sb.Bytes()
}

// Random generates a random string of the specified size using the specified charset.
// Play: https://go.dev/play/p/hX7F8StRq6Q
func Random[T any](size int, charset []rune) func(destination ro.Observable[T]) ro.Observable[[]byte] {
	if size <= 0 {
		panic("robytes.Random: Size parameter must be greater than 0")
	}
	if len(charset) <= 0 {
		panic("robytes.Random: Charset parameter must not be empty")
	}

	return ro.Map(
		func(value T) []byte {
			return random(size, charset)
		},
	)
}
