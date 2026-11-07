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
	"math/rand"
	"sync"
	"time"
)

//nolint:gochecknoglobals
var (
	globalRandMu sync.Mutex
	globalRand   = rand.New(rand.NewSource(time.Now().UnixNano())) //nolint:gosec
)

func randInt63() int64 {
	globalRandMu.Lock()
	defer globalRandMu.Unlock()
	return globalRand.Int63() //nolint:gosec
}

func randInt63n(n int64) int64 {
	globalRandMu.Lock()
	defer globalRandMu.Unlock()
	return globalRand.Int63n(n) //nolint:gosec
}

type Strategy func(attempt uint64) time.Duration

const maxDuration = time.Duration(math.MaxInt64)

func attemptIndex(attempt uint64) uint64 {
	if attempt == 0 {
		return 0
	}
	return attempt - 1
}

func clamp(d float64, maxDelay time.Duration) time.Duration {
	limit := maxDuration
	if maxDelay > 0 {
		limit = maxDelay
	}

	switch {
	case math.IsNaN(d) || d <= 0:
		return 0
	case d >= float64(limit):
		return limit
	default:
		return time.Duration(d)
	}
}

func Constant(delay time.Duration) Strategy {
	return func(uint64) time.Duration {
		return delay
	}
}

func Linear(initial, step, maxDelay time.Duration) Strategy {
	return func(attempt uint64) time.Duration {
		n := float64(attemptIndex(attempt))
		return clamp(float64(initial)+n*float64(step), maxDelay)
	}
}

func Exponential(initial time.Duration, factor float64, maxDelay time.Duration) Strategy {
	return func(attempt uint64) time.Duration {
		n := float64(attemptIndex(attempt))
		return clamp(float64(initial)*math.Pow(factor, n), maxDelay)
	}
}

func FullJitter(strategy Strategy) Strategy {
	return func(attempt uint64) time.Duration {
		d := strategy(attempt)
		if d <= 0 {
			return 0
		}
		if d == maxDuration {
			return time.Duration(randInt63())
		}
		return time.Duration(randInt63n(int64(d) + 1))
	}
}
