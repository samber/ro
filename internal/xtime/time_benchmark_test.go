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

package xtime

import (
	"syscall"
	"testing"
	"time"
	_ "unsafe" // required for runtime.nanotime
)

//
// This file aims to compare the performance of different implementations of internal stuff.
// For example: time.Now() vs syscall.Gettimeofday(), or std linked list vs custom.
//

var benchStartTime = time.Now()

//go:linkname nanotime runtime.nanotime
func nanotime() int64

// go test -benchmem -benchtime=100000000x -bench=Time ./bench/.
func BenchmarkDevelTime(b *testing.B) {
	b.Run("TimeGo", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = time.Now()
		}
	})

	// syscal.Gettimeofday is faster than time.Now()
	b.Run("TimeSyscallWallTime", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			var tv syscall.Timeval
			_ = syscall.Gettimeofday(&tv)
		}
	})

	// runtime.nanotime is faster than time.Now()
	b.Run("TimeRuntimeMonotonicTime", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = nanotime()
		}
	})

	// time.Since(benchStartTime) uses monotonic time
	b.Run("TimeRuntimeMonotonicTimeSince", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_ = time.Since(benchStartTime).Nanoseconds()
		}
	})
}
