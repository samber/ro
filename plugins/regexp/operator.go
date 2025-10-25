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


package roregexp

import (
	"regexp"

	"github.com/samber/ro"
)

// Find finds the first match of the pattern in the byte slice.
// Play: https://go.dev/play/p/9hM7n8kL5jU
func Find[T ~[]byte](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Map(func(v T) T {
		return pattern.Find(v)
	})
}

// FindString finds the first match of the pattern in the string.
func FindString[T ~string](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Map(func(v T) T {
		return T(pattern.FindString(string(v)))
	})
}

// FindSubmatch finds the first submatch of the pattern in the byte slice.
func FindSubmatch[T ~[]byte](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[[][]byte] {
	return ro.Map(func(v T) [][]byte {
		return pattern.FindSubmatch(v)
	})
}

// FindStringSubmatch finds the first submatch of the pattern in the string.
func FindStringSubmatch[T ~string](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[[]string] {
	return ro.Map(func(v T) []string {
		return pattern.FindStringSubmatch(string(v))
	})
}

// FindAll finds all matches of the pattern in the byte slice.
func FindAll[T ~[]byte](pattern *regexp.Regexp, n int) func(ro.Observable[T]) ro.Observable[[][]byte] {
	return ro.Map(func(v T) [][]byte {
		return pattern.FindAll(v, n)
	})
}

// FindAllString finds all matches of the pattern in the string.
func FindAllString[T ~string](pattern *regexp.Regexp, n int) func(ro.Observable[T]) ro.Observable[[]string] {
	return ro.Map(func(v T) []string {
		return pattern.FindAllString(string(v), n)
	})
}

// FindAllSubmatch finds all submatches of the pattern in the byte slice.
func FindAllSubmatch[T ~[]byte](pattern *regexp.Regexp, n int) func(ro.Observable[T]) ro.Observable[[][][]byte] {
	return ro.Map(func(v T) [][][]byte {
		return pattern.FindAllSubmatch(v, n)
	})
}

// FindAllStringSubmatch finds all submatches of the pattern in the string.
func FindAllStringSubmatch[T ~string](pattern *regexp.Regexp, n int) func(ro.Observable[T]) ro.Observable[[][]string] {
	return ro.Map(func(v T) [][]string {
		return pattern.FindAllStringSubmatch(string(v), n)
	})
}

// Match checks if the pattern matches the byte slice.
func Match[T ~[]byte](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[bool] {
	return ro.Map(func(v T) bool {
		return pattern.Match(v)
	})
}

// MatchString checks if the pattern matches the string.
func MatchString[T ~string](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[bool] {
	return ro.Map(func(v T) bool {
		return pattern.MatchString(string(v))
	})
}

// ReplaceAll replaces all matches of the pattern in the byte slice with the replacement.
func ReplaceAll[T ~[]byte](pattern *regexp.Regexp, repl T) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Map(func(v T) T {
		return pattern.ReplaceAll(v, repl)
	})
}

// ReplaceAllString replaces all matches of the pattern in the string with the replacement.
func ReplaceAllString[T ~string](pattern *regexp.Regexp, repl T) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Map(func(v T) T {
		return T(pattern.ReplaceAllString(string(v), string(repl)))
	})
}

// FilterMatch filters the byte slice if it matches the pattern.
func FilterMatch[T ~[]byte](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Filter(func(v T) bool {
		return pattern.Match(v)
	})
}

// FilterMatchString filters the string if it matches the pattern.
// Play: https://go.dev/play/p/9hM7n8kL5jU
func FilterMatchString[T ~string](pattern *regexp.Regexp) func(ro.Observable[T]) ro.Observable[T] {
	return ro.Filter(func(v T) bool {
		return pattern.MatchString(string(v))
	})
}
