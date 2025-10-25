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


package rostrconv

import (
	"strconv"

	"github.com/samber/ro"
)

// Atoi converts strings to integers using strconv.Atoi.
// Returns an error if the string cannot be converted to an integer.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"123", "456", "789"}), rostrconv.Atoi[string]()).
//	  Subscribe(ro.NewObserver[int](...))
// Play: https://go.dev/play/p/5hL9m8jK3nQ
func Atoi[T ~string]() func(ro.Observable[T]) ro.Observable[int] {
	return ro.MapErr(func(v T) (int, error) {
		return strconv.Atoi(string(v))
	})
}

// ParseInt converts strings to int64 values with specified base and bit size.
// The base parameter determines the number system (e.g., 10 for decimal, 16 for hexadecimal).
// The bitSize parameter specifies the integer type size (e.g., 32 for int32, 64 for int64).
// Returns an error if the string cannot be converted to the specified integer type.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"123", "FF", "1010"}), rostrconv.ParseInt[string](16, 64)). // Parse as hex, 64-bit
//	  Subscribe(ro.NewObserver[int64](...))
// Play: https://go.dev/play/p/CqjCmQVAPXC
func ParseInt[T ~string](base int, bitSize int) func(ro.Observable[T]) ro.Observable[int64] {
	return ro.MapErr(func(v T) (int64, error) {
		return strconv.ParseInt(string(v), base, bitSize)
	})
}

// ParseFloat converts strings to float64 values with specified bit size.
// The bitSize parameter specifies the float type size (e.g., 32 for float32, 64 for float64).
// Returns an error if the string cannot be converted to a float.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"3.14", "2.718", "1.414"}), rostrconv.ParseFloat[string](64)). // Parse as 64-bit float
//	  Subscribe(ro.NewObserver[float64](...))
// Play: https://go.dev/play/p/g-YvtjXtX7V
func ParseFloat[T ~string](bitSize int) func(ro.Observable[T]) ro.Observable[float64] {
	return ro.MapErr(func(v T) (float64, error) {
		return strconv.ParseFloat(string(v), bitSize)
	})
}

// ParseBool converts strings to boolean values using strconv.ParseBool.
// Accepts "1", "t", "T", "true", "TRUE", "True" for true values.
// Accepts "0", "f", "F", "false", "FALSE", "False" for false values.
// Returns an error if the string cannot be converted to a boolean.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"true", "false", "1", "0"}), rostrconv.ParseBool[string]()).
//	  Subscribe(ro.NewObserver[bool](...))
// Play: https://go.dev/play/p/2C5fkrRLyW_g
func ParseBool[T ~string]() func(ro.Observable[T]) ro.Observable[bool] {
	return ro.MapErr(func(v T) (bool, error) {
		return strconv.ParseBool(string(v))
	})
}

// ParseUint converts strings to uint64 values with specified base and bit size.
// The base parameter determines the number system (e.g., 10 for decimal, 16 for hexadecimal).
// The bitSize parameter specifies the integer type size (e.g., 32 for uint32, 64 for uint64).
// Returns an error if the string cannot be converted to the specified unsigned integer type.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"123", "FF", "1010"}), rostrconv.ParseUint[string](16, 64)). // Parse as hex, 64-bit unsigned
//	  Subscribe(ro.NewObserver[uint64](...))
func ParseUint[T ~string](base int, bitSize int) func(ro.Observable[T]) ro.Observable[uint64] {
	return ro.MapErr(func(v T) (uint64, error) {
		return strconv.ParseUint(string(v), base, bitSize)
	})
}

// ParseUint64 is an alias for ParseUint that specifically returns uint64.
// It provides the same functionality as ParseUint with bitSize=64.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"123", "456", "789"}), rostrconv.ParseUint64[string](10, 64)). // Parse as decimal, 64-bit unsigned
//	  Subscribe(ro.NewObserver[uint64](...))
func ParseUint64[T ~string](base int, bitSize int) func(ro.Observable[T]) ro.Observable[uint64] {
	return ro.MapErr(func(v T) (uint64, error) {
		return strconv.ParseUint(string(v), base, bitSize)
	})
}

// FormatBool converts boolean values to strings using strconv.FormatBool.
// Returns "true" for true values and "false" for false values.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]bool{true, false, true}), rostrconv.FormatBool()).
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/8vDdaQyzoi_b
func FormatBool() func(ro.Observable[bool]) ro.Observable[string] {
	return ro.Map(strconv.FormatBool)
}

// FormatFloat converts float64 values to strings with specified format, precision, and bit size.
// The format parameter (mt) can be:
//
//	'f' for fixed-point notation
//	'e' for scientific notation
//	'g' for the shortest representation
//	'x' for hexadecimal notation
//
// The prec parameter specifies the precision (number of digits).
// The bitSize parameter specifies the float type size (32 or 64).
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]float64{3.14159, 2.71828}), rostrconv.FormatFloat('f', 2, 64)). // Fixed-point, 2 decimal places
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/GWSPE4Mp-uy
func FormatFloat(mt byte, prec, bitSize int) func(ro.Observable[float64]) ro.Observable[string] {
	return ro.Map(func(v float64) string {
		return strconv.FormatFloat(v, mt, prec, bitSize)
	})
}

// FormatComplex converts complex128 values to strings with specified format, precision, and bit size.
// The format parameter (mt) can be:
//
//	'f' for fixed-point notation
//	'e' for scientific notation
//	'g' for the shortest representation
//
// The prec parameter specifies the precision (number of digits).
// The bitSize parameter specifies the float type size (64 or 128).
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]complex128{3+4i, 1+2i}), rostrconv.FormatComplex('f', 2, 128)). // Fixed-point, 2 decimal places
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/gbp_kl7XJWv
func FormatComplex(mt byte, prec, bitSize int) func(ro.Observable[complex128]) ro.Observable[string] {
	return ro.Map(func(v complex128) string {
		return strconv.FormatComplex(v, mt, prec, bitSize)
	})
}

// FormatInt converts int64 values to strings with specified base.
// The base parameter determines the number system (e.g., 10 for decimal, 16 for hexadecimal).
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]int64{123, 456, 789}), rostrconv.FormatInt[string](16)). // Format as hexadecimal
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/hUpBVHRJgXY
func FormatInt[T ~string](base int) func(ro.Observable[int64]) ro.Observable[string] {
	return ro.Map(func(v int64) string {
		return strconv.FormatInt(v, base)
	})
}

// FormatUint converts uint64 values to strings with specified base.
// The base parameter determines the number system (e.g., 10 for decimal, 16 for hexadecimal).
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]uint64{123, 456, 789}), rostrconv.FormatUint[string](16)). // Format as hexadecimal
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/h4TYG9sFPZw
func FormatUint[T ~string](base int) func(ro.Observable[uint64]) ro.Observable[string] {
	return ro.Map(func(v uint64) string {
		return strconv.FormatUint(v, base)
	})
}

// Itoa converts integers to strings using strconv.Itoa.
// This is equivalent to FormatInt with base 10.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]int{123, 456, 789}), rostrconv.Itoa()).
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/6hN7k9jL4mR
func Itoa() func(ro.Observable[int]) ro.Observable[string] {
	return ro.Map(strconv.Itoa)
}

// Quote converts strings to Go string literals using strconv.Quote.
// This adds double quotes and escapes special characters.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{"hello", "world\n", "test"}), rostrconv.Quote()).
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/O72Y-oUwBxr
func Quote() func(ro.Observable[string]) ro.Observable[string] {
	return ro.Map(strconv.Quote)
}

// QuoteRune converts runes to Go character literals using strconv.QuoteRune.
// This adds single quotes and escapes special characters.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]rune{'a', 'b', 'c'}), rostrconv.QuoteRune()).
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/tCnviYGuSMn
func QuoteRune() func(ro.Observable[rune]) ro.Observable[string] {
	return ro.Map(strconv.QuoteRune)
}

// Unquote converts Go string literals back to strings using strconv.Unquote.
// This removes quotes and unescapes special characters.
// Returns an error if the string is not a valid Go string literal.
//
// Example:
//
//	ro.Pipe(ro.FromSlice([]string{`"hello"`, `"world\n"`, `"test"`}), rostrconv.Unquote()).
//	  Subscribe(ro.NewObserver[string](...))
// Play: https://go.dev/play/p/cMaHM-He8NT
func Unquote() func(ro.Observable[string]) ro.Observable[string] {
	return ro.MapErr(strconv.Unquote)
}
