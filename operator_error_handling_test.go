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

package ro

import (
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestOperatorErrorHandlingCatch(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			Catch(func(err error) Observable[int] {
				is.Fail("never")
				return Empty[int]()
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			Catch(func(err error) Observable[int] {
				is.EqualError(err, "ro.Observer: "+assert.AnError.Error())
				return Of(4, 5, 6)
			}),
		),
	)
	is.Equal([]int{1, 2, 4, 5, 6}, values)
	is.NoError(err)
}

func TestOperatorErrorHandlingOnErrorResumeNextWith(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		OnErrorResumeNextWith(
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(4)
				observer.Error(assert.AnError)

				return nil
			}),
			Of(5, 6, 7),
		)(Of(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3, 4, 5, 6, 7}, values)
	is.NoError(err)

	values, err = Collect(
		OnErrorResumeNextWith(
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(2)
				observer.Error(errors.New("error 3"))

				return nil
			}),
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(3)
				observer.Error(errors.New("error 2"))

				return nil
			}),
		)(
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(1)
				observer.Error(errors.New("error 1"))

				return nil
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.Errorf(err, "error 3")

	values, err = Collect(
		OnErrorResumeNextWith(
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(2)
				observer.Error(errors.New("error 3"))

				return nil
			}),
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(3)
				observer.Complete()

				return nil
			}),
		)(
			NewObservable(func(observer Observer[int]) Teardown {
				observer.Next(1)
				observer.Error(errors.New("error 1"))

				return nil
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			OnErrorResumeNextWith(Of(4, 5, 6)),
		),
	)
	is.Equal([]int{1, 2, 4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		OnErrorResumeNextWith[int]()(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		OnErrorResumeNextWith[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorErrorHandlingOnErrorReturn(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			OnErrorReturn(4),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			OnErrorReturn(4),
		),
	)
	is.Equal([]int{1, 2, 4}, values)
	is.NoError(err)
}

func TestOperatorErrorHandlingRetry(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			Retry[int](),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	crash := 0
	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 && crash < 2 {
					crash++

					panic(assert.AnError)
				}

				return x
			}),
			Retry[int](),
		),
	)
	is.Equal([]int{1, 2, 1, 2, 1, 2, 3}, values)
	is.NoError(err)
}

func TestOperatorErrorHandlingRetryWithConfig(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 400*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			RetryWithConfig[int](RetryConfig{}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	crash := 0
	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 && crash < 2 {
					crash++

					panic(assert.AnError)
				}

				return x
			}),
			RetryWithConfig[int](RetryConfig{}),
		),
	)
	is.Equal([]int{1, 2, 1, 2, 1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			RetryWithConfig[int](RetryConfig{MaxRetries: 2}),
		),
	)
	is.Equal([]int{1, 2, 1, 2, 1, 2}, values)
	is.EqualError(err, "ro.Observer: "+assert.AnError.Error())

	start := time.Now()
	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			RetryWithConfig[int](RetryConfig{MaxRetries: 2, Delay: 75 * time.Millisecond}),
		),
	)
	is.Equal([]int{1, 2, 1, 2, 1, 2}, values)
	is.WithinDuration(time.Now(), start.Add(150*time.Millisecond), 30*time.Millisecond)
	is.EqualError(err, "ro.Observer: "+assert.AnError.Error())

	total := 0
	values, err = Collect(
		Pipe2(
			Of(1, 2, 3),
			Map(func(x int) int {
				total++
				if total > 10 {
					panic(assert.AnError)
				}

				if x == 3 {
					panic(assert.AnError)
				}

				return x
			}),
			RetryWithConfig[int](RetryConfig{MaxRetries: 2, ResetOnSuccess: true}),
		),
	)
	is.Equal([]int{1, 2, 1, 2, 1, 2, 1}, values)
	is.EqualError(err, "ro.Observer: "+assert.AnError.Error())
}

func TestOperatorErrorHandlingThrowIfEmpty(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			ThrowIfEmpty[int](func() error {
				is.Fail("never")
				return assert.AnError
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int](),
			ThrowIfEmpty[int](func() error {
				return assert.AnError
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			ThrowIfEmpty[int](func() error {
				is.Fail("never")
				return assert.AnError
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorErrorHandlingDoWhile(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	counter := int32(0)
	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			DoWhile[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 2
			}),
		),
	)
	is.Equal([]int{1, 2, 3, 1, 2, 3}, values)
	is.NoError(err)

	atomic.StoreInt32(&counter, 0)

	values, err = Collect(
		Pipe1(
			Empty[int](),
			DoWhile[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 2
			}),
		),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	atomic.StoreInt32(&counter, 0)

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			DoWhile[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 2
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorErrorHandlingWhile(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	counter := int32(0)
	values, err := Collect(
		Pipe1(
			Of(1, 2, 3),
			While[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 2
			}),
		),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	atomic.StoreInt32(&counter, 0)

	values, err = Collect(
		Pipe1(
			Empty[int](),
			While[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 2
			}),
		),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			While[int](func() bool {
				return false
			}),
		),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	atomic.StoreInt32(&counter, 0)

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			While[int](func() bool {
				return atomic.AddInt32(&counter, 1) < 3
			}),
		),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}
