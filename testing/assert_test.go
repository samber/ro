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

package rotesting

import (
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

//nolint:paralleltest,tparallel // subtests depend on wall-clock sleeps and must not run concurrently
func TestAssert(t *testing.T) {
	t.Parallel()

	t.Run("initial state", func(t *testing.T) {
		is := assert.New(t)

		a, ok := Assert[int](t).(*assertImpl[int])
		is.True(ok)
		is.Equal(t, t)
		is.Empty(a.assertions)
		is.Nil(a.source)
	})

	t.Run("complete normally", func(t *testing.T) {
		Assert[int](t).
			Source(ro.Just(1, 2, 3)).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectComplete().
			Verify()
	})

	t.Run("complete on error", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				destination.Next(1)
				destination.Next(2)
				destination.Next(3)
				destination.Error(assert.AnError)
				destination.Next(4)
				return nil
			})).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectError(assert.AnError).
			Verify()
	})

	t.Run("async observable", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					destination.Next(1)
					destination.Next(2)
					destination.Next(3)
					destination.Error(assert.AnError)
					destination.Next(4)
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectNext(2).
			ExpectNext(3).
			ExpectError(assert.AnError).
			Verify()
	})

	t.Run("duration epsilon", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					time.Sleep(10 * time.Millisecond)
					destination.Next(1)
					time.Sleep(10 * time.Millisecond)
					destination.Next(2)
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectDurationEpsilon(10*time.Millisecond, 5*time.Millisecond).
			ExpectNext(2).
			Verify()
	})

	t.Run("duration less than", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					time.Sleep(5 * time.Millisecond)
					destination.Next(1)
					time.Sleep(5 * time.Millisecond)
					destination.Next(2)
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectDurationLessThan(20 * time.Millisecond).
			ExpectNext(2).
			Verify()
	})

	t.Run("duration greater than", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					time.Sleep(15 * time.Millisecond)
					destination.Next(1)
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectDurationGreaterThan(10 * time.Millisecond).
			Verify()
	})

	t.Run("duration in range", func(t *testing.T) {
		Assert[int](t).
			Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
				go func() {
					time.Sleep(10 * time.Millisecond)
					destination.Next(1)
					time.Sleep(10 * time.Millisecond)
					destination.Next(2)
				}()
				return nil
			})).
			ExpectNext(1).
			ExpectDurationInRange(5*time.Millisecond, 20*time.Millisecond).
			ExpectNext(2).
			Verify()
	})
}
