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


package roratelimit

import (
	"time"

	"github.com/samber/ro"
)

// NewRateLimiter creates a rate limiter that allows count items per interval for each key.
// Play: https://go.dev/play/p/YNhnGgrMWmj
func NewRateLimiter[T any](count int64, interval time.Duration, keyGetter func(T) string) func(destination ro.Observable[T]) ro.Observable[T] {
	return func(source ro.Observable[T]) ro.Observable[T] {
		return ro.Pipe2(
			source,
			ro.GroupBy(keyGetter),
			ro.MergeMap(
				ro.PipeOp3(
					ro.WindowWhen[T](ro.Interval(interval)),
					ro.Map(ro.Take[T](count)),
					ro.MergeAll[T](),
				),
			),
		)
	}
}
