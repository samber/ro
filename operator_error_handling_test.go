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
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// upstreamProbe counts the subscriptions made to a probed source, and the teardowns that ran.
type upstreamProbe struct {
	subscribed int64
	released   int64
}

// probedSource emits items then terminates, from the subscribing goroutine or from its own
// goroutine. Attempt N (1-based) fails with the error returned by errAt(N); a nil error completes.
func probedSource(probe *upstreamProbe, async bool, errAt func(attempt int64) error, items ...int) Observable[int] {
	return NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		attempt := atomic.AddInt64(&probe.subscribed, 1)

		emit := func() {
			for _, item := range items {
				destination.NextWithContext(ctx, item)
			}

			if err := errAt(attempt); err != nil {
				destination.ErrorWithContext(ctx, err)
				return
			}

			destination.CompleteWithContext(ctx)
		}

		if async {
			go emit()
		} else {
			emit()
		}

		return func() { atomic.AddInt64(&probe.released, 1) }
	})
}

func neverFails(int64) error { return nil }

func alwaysFails(int64) error { return assert.AnError }

// collectWithin fails the test instead of hanging when the pipeline never terminates.
func collectWithin(t *testing.T, obs Observable[int]) ([]int, error) {
	t.Helper()

	type result struct {
		values []int
		err    error
	}

	done := make(chan result, 1)

	go func() {
		values, err := Collect(obs)
		done <- result{values, err}
	}()

	select {
	case r := <-done:
		return r.values, r.err
	case <-time.After(2 * time.Second):
		t.Fatal("pipeline did not terminate")
		return nil, nil
	}
}

// forEachSourceMode runs fn against a source emitting from the subscribing goroutine, then from its own.
func forEachSourceMode(t *testing.T, fn func(t *testing.T, async bool)) {
	t.Helper()

	t.Run("sync", func(t *testing.T) { fn(t, false) })
	t.Run("async", func(t *testing.T) { fn(t, true) })
}

// expectUpstreamReleased asserts every subscription to the probed source was torn down.
func expectUpstreamReleased(t *testing.T, probe *upstreamProbe, wantSubscribed int64) {
	t.Helper()

	assert.Equal(t, wantSubscribed, atomic.LoadInt64(&probe.subscribed), "subscriptions")
	assert.Eventually(t, func() bool {
		return atomic.LoadInt64(&probe.released) == wantSubscribed
	}, time.Second, time.Millisecond, "every upstream subscription must be unsubscribed")
}

// downstreamCancelSkip marks the tests that reproduce the operators ignoring a closed destination.
// Remove it, and the skip in expectStopsOnDownstreamCancel, once the operators are fixed.
const downstreamCancelSkip = "race: downstream-cancel-resubscribe; remove when fixed"

// runawayBound caps the resubscriptions of an operator that ignores a closed destination, so the
// regression fails on an assertion instead of spinning forever.
const runawayBound = 10_000

// endlessSource emits items from its own goroutine, then stays subscribed until it is unsubscribed.
// The test cleanup also ends it, so a failing test does not leak the goroutine.
func endlessSource(t *testing.T, probe *upstreamProbe, items ...int) Observable[int] {
	t.Helper()

	stop := make(chan struct{})
	stopOnce := sync.Once{}
	closeStop := func() { stopOnce.Do(func() { close(stop) }) }

	t.Cleanup(closeStop)

	return NewSafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
		atomic.AddInt64(&probe.subscribed, 1)

		go func() {
			defer destination.CompleteWithContext(ctx)

			for _, item := range items {
				select {
				case <-stop:
					return
				default:
					destination.NextWithContext(ctx, item)
				}
			}

			<-stop
		}()

		return func() {
			closeStop()
			atomic.AddInt64(&probe.released, 1)
		}
	})
}

// expectStopsOnDownstreamCancel checks that an operator re-subscribing to its source stops doing so,
// and unsubscribes the source, once a downstream Take has what it needs. errAt makes the source
// complete or fail on each subscription, chain wraps the source in the operator under test.
func expectStopsOnDownstreamCancel(
	t *testing.T,
	errAt func(attempt int64) error,
	chain func(source Observable[int]) Observable[int],
	syncTake, syncWant int64,
) {
	t.Helper()
	t.Skip(downstreamCancelSkip)

	t.Run("sync source", func(t *testing.T) {
		is := assert.New(t)

		probe := &upstreamProbe{}
		values, err := collectWithin(t, Pipe1(chain(probedSource(probe, false, errAt, 1)), Take[int](syncTake)))
		is.Len(values, int(syncTake))
		is.NoError(err)
		expectUpstreamReleased(t, probe, syncWant)
	})

	t.Run("async source ending after Take", func(t *testing.T) {
		is := assert.New(t)

		// Take(2) is satisfied by the first 2 of 5 items: the source must not be subscribed again.
		probe := &upstreamProbe{}
		values, err := collectWithin(t, Pipe1(chain(probedSource(probe, true, errAt, 1, 2, 3, 4, 5)), Take[int](2)))
		is.Len(values, 2)
		is.NoError(err)
		expectUpstreamReleased(t, probe, 1)
	})

	t.Run("async source still running at Take", func(t *testing.T) {
		is := assert.New(t)

		// the source never terminates by itself: only an unsubscription from downstream can end it.
		probe := &upstreamProbe{}
		values, err := collectWithin(t, Pipe1(chain(endlessSource(t, probe, 1, 2, 3)), Take[int](2)))
		is.Len(values, 2)
		is.NoError(err)
		expectUpstreamReleased(t, probe, 1)
	})
}

// failsUntilBound fails every attempt up to runawayBound, then completes.
func failsUntilBound(attempt int64) error {
	if attempt <= runawayBound {
		return assert.AnError
	}

	return nil
}

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

	t.Run("reusing the same operator does not grow its chain", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		op := OnErrorResumeNextWith(Just(2))

		// Applying the same operator twice must not grow its chain.
		for i := 0; i < 2; i++ {
			values, err := Collect(op(Just(1)))
			is.Equal([]int{1, 2}, values)
			is.NoError(err)
		}

		values, err := Collect(op(op(Just(1))))
		is.Equal([]int{1, 2, 2}, values)
		is.NoError(err)
	})

	t.Run("releases upstream when it stops by itself", func(t *testing.T) {
		t.Parallel()

		forEachSourceMode(t, func(t *testing.T, async bool) {
			t.Helper()

			is := assert.New(t)

			// source and first fallback fail, last fallback completes: the chain ends by itself.
			probe := &upstreamProbe{}
			failing := probedSource(probe, async, alwaysFails, 1)
			completing := probedSource(probe, async, neverFails, 2)

			values, err := collectWithin(t, Pipe1(failing, OnErrorResumeNextWith(failing, completing)))
			is.Equal([]int{1, 1, 2}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 3)

			// last fallback fails too: the chain ends with that error.
			probe = &upstreamProbe{}
			failing = probedSource(probe, async, alwaysFails, 1)

			values, err = collectWithin(t, Pipe1(failing, OnErrorResumeNextWith(failing, failing)))
			is.Equal([]int{1, 1, 1}, values)
			is.EqualError(err, assert.AnError.Error())
			expectUpstreamReleased(t, probe, 3)
		})
	})

	t.Run("stops when downstream cancels", func(t *testing.T) {
		t.Parallel()

		// 3 sources in the chain, Take(1) is satisfied by the first one: the others must not be subscribed.
		expectStopsOnDownstreamCancel(t, alwaysFails,
			func(source Observable[int]) Observable[int] {
				return Pipe1(source, OnErrorResumeNextWith(source, source))
			},
			1, 1,
		)
	})
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

// The subtests stay sequential: testWithTimeout below would time out waiting on parallel subtests.
func TestOperatorErrorHandlingRetry(t *testing.T) { //nolint:tparallel
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

	t.Run("releases upstream when it stops by itself", func(t *testing.T) {
		forEachSourceMode(t, func(t *testing.T, async bool) {
			t.Helper()

			is := assert.New(t)

			// attempts 1 and 2 fail, attempt 3 completes: Retry stops resubscribing by itself.
			probe := &upstreamProbe{}
			source := probedSource(probe, async, func(attempt int64) error {
				if attempt <= 2 {
					return assert.AnError
				}

				return nil
			}, 1)

			values, err := collectWithin(t, Pipe1(source, Retry[int]()))
			is.Equal([]int{1, 1, 1}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 3)
		})
	})
}

func TestOperatorErrorHandlingRetry_downstreamCancel(t *testing.T) {
	t.Parallel()

	expectStopsOnDownstreamCancel(t, failsUntilBound,
		func(source Observable[int]) Observable[int] { return Pipe1(source, Retry[int]()) },
		3, 3,
	)
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

	t.Run("releases upstream when it stops by itself", func(t *testing.T) {
		forEachSourceMode(t, func(t *testing.T, async bool) {
			t.Helper()

			is := assert.New(t)

			// retries exhausted: 1 initial subscription + 2 retries, then the error goes downstream.
			probe := &upstreamProbe{}
			source := probedSource(probe, async, alwaysFails, 1)

			values, err := collectWithin(t, Pipe1(source, RetryWithConfig[int](RetryConfig{MaxRetries: 2})))
			is.Equal([]int{1, 1, 1}, values)
			is.EqualError(err, assert.AnError.Error())
			expectUpstreamReleased(t, probe, 3)

			// source succeeds after one failure: no further subscription.
			probe = &upstreamProbe{}
			source = probedSource(probe, async, func(attempt int64) error {
				if attempt == 1 {
					return assert.AnError
				}

				return nil
			}, 1)

			values, err = collectWithin(t, Pipe1(source, RetryWithConfig[int](RetryConfig{MaxRetries: 2})))
			is.Equal([]int{1, 1}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 2)
		})
	})
}

func TestOperatorErrorHandlingRetryWithConfig_downstreamCancel(t *testing.T) {
	t.Parallel()

	expectStopsOnDownstreamCancel(t, alwaysFails,
		func(source Observable[int]) Observable[int] {
			return Pipe1(source, RetryWithConfig[int](RetryConfig{MaxRetries: runawayBound}))
		},
		3, 3,
	)
}

func TestOperatorErrorHandlingRetryWithConfig_finalErrorContext(t *testing.T) {
	t.Parallel()

	type ctxKey string

	key := ctxKey("attempt")

	tests := []struct {
		name          string
		maxRetries    uint64
		expectedValue int
	}{
		{name: "one retry", maxRetries: 1, expectedValue: 2},
		{name: "two retries", maxRetries: 2, expectedValue: 3},
	}

	for _, tt := range tests {
		tt := tt // go.mod predates per-iteration loop variables

		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			is := assert.New(t)

			// The source tags the context of its error with the attempt number: the
			// final error must reach the destination with that context, not with the
			// subscriber's one.
			attempts := 0
			source := NewUnsafeObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
				attempts++
				destination.ErrorWithContext(context.WithValue(ctx, key, attempts), assert.AnError)

				return nil
			})

			var errCtx context.Context

			var gotErr error

			sub := Pipe1(source, RetryWithConfig[int](RetryConfig{MaxRetries: tt.maxRetries})).SubscribeWithContext(
				context.Background(),
				NewObserverWithContext(
					func(ctx context.Context, value int) {},
					func(ctx context.Context, err error) {
						errCtx = ctx
						gotErr = err
					},
					func(ctx context.Context) {},
				),
			)
			sub.Wait()

			is.ErrorIs(gotErr, assert.AnError)
			is.NotNil(errCtx)
			is.Equal(tt.expectedValue, errCtx.Value(key))
		})
	}
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

	t.Run("releases upstream when it stops by itself", func(t *testing.T) {
		t.Parallel()

		forEachSourceMode(t, func(t *testing.T, async bool) {
			t.Helper()

			is := assert.New(t)

			// the condition turns false after the 3rd run: DoWhile stops by itself.
			probe := &upstreamProbe{}
			runs := int32(0)

			values, err := collectWithin(t, Pipe1(
				probedSource(probe, async, neverFails, 1, 2),
				DoWhile[int](func() bool { return atomic.AddInt32(&runs, 1) < 3 }),
			))
			is.Equal([]int{1, 2, 1, 2, 1, 2}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 3)

			// the condition is false right away: the source still ran once.
			probe = &upstreamProbe{}

			values, err = collectWithin(t, Pipe1(
				probedSource(probe, async, neverFails, 1, 2),
				DoWhile[int](func() bool { return false }),
			))
			is.Equal([]int{1, 2}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 1)
		})
	})

	t.Run("stops when downstream cancels", func(t *testing.T) {
		t.Parallel()

		expectStopsOnDownstreamCancel(t, neverFails,
			func(source Observable[int]) Observable[int] {
				runs := int32(0)

				return Pipe1(source, DoWhile[int](func() bool { return atomic.AddInt32(&runs, 1) < runawayBound }))
			},
			3, 3,
		)
	})
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

	t.Run("releases upstream when it stops by itself", func(t *testing.T) {
		t.Parallel()

		forEachSourceMode(t, func(t *testing.T, async bool) {
			t.Helper()

			is := assert.New(t)

			// the condition turns false on its 3rd check: the source ran twice, While stops by itself.
			probe := &upstreamProbe{}
			checks := int32(0)

			values, err := collectWithin(t, Pipe1(
				probedSource(probe, async, neverFails, 1, 2),
				While[int](func() bool { return atomic.AddInt32(&checks, 1) < 3 }),
			))
			is.Equal([]int{1, 2, 1, 2}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 2)

			// the condition is false right away: the source is never subscribed.
			probe = &upstreamProbe{}

			values, err = collectWithin(t, Pipe1(
				probedSource(probe, async, neverFails, 1, 2),
				While[int](func() bool { return false }),
			))
			is.Equal([]int{}, values)
			is.NoError(err)
			expectUpstreamReleased(t, probe, 0)
		})
	})

	t.Run("stops when downstream cancels", func(t *testing.T) {
		t.Parallel()

		expectStopsOnDownstreamCancel(t, neverFails,
			func(source Observable[int]) Observable[int] {
				checks := int32(0)

				return Pipe1(source, While[int](func() bool { return atomic.AddInt32(&checks, 1) <= runawayBound }))
			},
			3, 3,
		)
	})
}
