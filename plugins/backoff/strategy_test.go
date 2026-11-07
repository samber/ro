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

package robackoff

import (
	"math"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// --- internal helpers ---

func TestAttemptIndex(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.Equal(uint64(0), attemptIndex(0))
	is.Equal(uint64(0), attemptIndex(1))
	is.Equal(uint64(1), attemptIndex(2))
	is.Equal(uint64(9), attemptIndex(10))
}

func TestClamp(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// NaN → 0
	is.Equal(time.Duration(0), clamp(math.NaN(), 0))

	// negative / zero → 0
	is.Equal(time.Duration(0), clamp(-1, 0))
	is.Equal(time.Duration(0), clamp(0, 0))

	// normal value, no cap
	is.Equal(100*time.Millisecond, clamp(float64(100*time.Millisecond), 0))

	// value exceeds cap → cap
	is.Equal(500*time.Millisecond, clamp(float64(time.Second), 500*time.Millisecond))

	// value equals cap exactly → cap
	is.Equal(500*time.Millisecond, clamp(float64(500*time.Millisecond), 500*time.Millisecond))

	// value within cap → value
	is.Equal(200*time.Millisecond, clamp(float64(200*time.Millisecond), 500*time.Millisecond))

	// very large value, no cap → maxDuration
	is.Equal(maxDuration, clamp(math.MaxFloat64, 0))
}

// --- Constant ---

func TestStrategyConstant(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	s := Constant(200 * time.Millisecond)
	// same value regardless of attempt
	for _, attempt := range []uint64{0, 1, 2, 10, 100} {
		is.Equal(200*time.Millisecond, s(attempt))
	}
}

func TestStrategyConstantZero(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	s := Constant(0)
	is.Equal(time.Duration(0), s(1))
	is.Equal(time.Duration(0), s(99))
}

func TestStrategyConstantNegative(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Negative delay is returned as-is (no clamp — Constant is a direct pass-through).
	s := Constant(-1 * time.Second)
	is.Equal(-1*time.Second, s(1))
}

// --- Linear ---

func TestStrategyLinear(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// initial=100ms, step=50ms, no cap
	s := Linear(100*time.Millisecond, 50*time.Millisecond, 0)

	// attempt 1 → index 0 → 100 + 0*50 = 100ms
	is.Equal(100*time.Millisecond, s(1))
	// attempt 2 → index 1 → 100 + 1*50 = 150ms
	is.Equal(150*time.Millisecond, s(2))
	// attempt 3 → index 2 → 100 + 2*50 = 200ms
	is.Equal(200*time.Millisecond, s(3))
	// attempt 5 → index 4 → 100 + 4*50 = 300ms
	is.Equal(300*time.Millisecond, s(5))
}

func TestStrategyLinearAttemptZero(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// attempt 0 → attemptIndex(0)=0 → initial + 0*step = initial
	s := Linear(100*time.Millisecond, 50*time.Millisecond, 0)
	is.Equal(100*time.Millisecond, s(0))
}

func TestStrategyLinearCap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	cap := 250 * time.Millisecond
	s := Linear(100*time.Millisecond, 50*time.Millisecond, cap)

	is.Equal(100*time.Millisecond, s(1)) // 100ms < cap
	is.Equal(150*time.Millisecond, s(2)) // 150ms < cap
	is.Equal(200*time.Millisecond, s(3)) // 200ms < cap
	is.Equal(cap, s(4))                  // 250ms == cap
	is.Equal(cap, s(5))                  // 300ms > cap → clamped
	is.Equal(cap, s(100))                // well over cap
}

func TestStrategyLinearZeroStep(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// step=0 behaves like Constant
	s := Linear(300*time.Millisecond, 0, 0)
	is.Equal(300*time.Millisecond, s(1))
	is.Equal(300*time.Millisecond, s(10))
}

// --- Exponential ---

func TestStrategyExponential(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// initial=100ms, factor=2, no cap
	s := Exponential(100*time.Millisecond, 2.0, 0)

	// attempt 1 → index 0 → 100 * 2^0 = 100ms
	is.Equal(100*time.Millisecond, s(1))
	// attempt 2 → index 1 → 100 * 2^1 = 200ms
	is.Equal(200*time.Millisecond, s(2))
	// attempt 3 → index 2 → 100 * 2^2 = 400ms
	is.Equal(400*time.Millisecond, s(3))
	// attempt 4 → index 3 → 100 * 2^3 = 800ms
	is.Equal(800*time.Millisecond, s(4))
}

func TestStrategyExponentialAttemptZero(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	s := Exponential(100*time.Millisecond, 2.0, 0)
	// attempt 0 → index 0 → 100 * 2^0 = 100ms
	is.Equal(100*time.Millisecond, s(0))
}

func TestStrategyExponentialCap(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	cap := 500 * time.Millisecond
	s := Exponential(100*time.Millisecond, 2.0, cap)

	is.Equal(100*time.Millisecond, s(1)) // 100ms
	is.Equal(200*time.Millisecond, s(2)) // 200ms
	is.Equal(400*time.Millisecond, s(3)) // 400ms
	is.Equal(cap, s(4))                  // 800ms → clamped to 500ms
	is.Equal(cap, s(10))                 // well over cap
}

func TestStrategyExponentialFactorOne(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// factor=1 behaves like Constant
	s := Exponential(200*time.Millisecond, 1.0, 0)
	is.Equal(200*time.Millisecond, s(1))
	is.Equal(200*time.Millisecond, s(5))
}

// --- FullJitter ---

func TestStrategyFullJitter(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	base := Constant(1 * time.Second)
	s := FullJitter(base)

	// Run many samples: all must be in [0, 1s]
	for i := uint64(0); i < 1000; i++ {
		d := s(1)
		is.GreaterOrEqual(d, time.Duration(0))
		is.LessOrEqual(d, time.Second)
	}
}

func TestStrategyFullJitterZeroBase(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// When the wrapped strategy returns 0, jitter should also return 0.
	s := FullJitter(Constant(0))
	for i := uint64(0); i < 100; i++ {
		is.Equal(time.Duration(0), s(i))
	}
}

func TestStrategyFullJitterNegativeBase(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// A base returning a negative value is treated as ≤ 0 → 0.
	s := FullJitter(func(uint64) time.Duration { return -1 * time.Second })
	is.Equal(time.Duration(0), s(1))
}

func TestStrategyFullJitterMaxDuration(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// When base == maxDuration the result must be a valid non-negative duration.
	s := FullJitter(func(uint64) time.Duration { return maxDuration })
	for i := 0; i < 100; i++ {
		d := s(1)
		is.GreaterOrEqual(d, time.Duration(0))
	}
}

func TestStrategyFullJitterDistribution(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// With enough samples the jitter should produce both small and large values,
	// confirming it's not degenerate (not always 0 or always max).
	const n = 500
	base := Constant(100 * time.Millisecond)
	s := FullJitter(base)

	var below50, above50 int
	for i := 0; i < n; i++ {
		if s(1) < 50*time.Millisecond {
			below50++
		} else {
			above50++
		}
	}
	// With a uniform distribution over [0,100ms] we expect ~50% each side.
	// Allow a generous margin to avoid flakiness.
	is.Greater(below50, n/5, "expected some values below 50ms")
	is.Greater(above50, n/5, "expected some values above 50ms")
}

// --- FullJitter wrapping Exponential ---

func TestStrategyFullJitterExponential(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	s := FullJitter(Exponential(100*time.Millisecond, 2.0, time.Second))

	// attempt 1: base = 100ms → jitter in [0, 100ms]
	for i := 0; i < 200; i++ {
		d := s(1)
		is.GreaterOrEqual(d, time.Duration(0))
		is.LessOrEqual(d, 100*time.Millisecond)
	}

	// attempt 4: base capped at 1s → jitter in [0, 1s]
	for i := 0; i < 200; i++ {
		d := s(4)
		is.GreaterOrEqual(d, time.Duration(0))
		is.LessOrEqual(d, time.Second)
	}
}

