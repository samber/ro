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
	"fmt"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func zipCompletionVariants() []struct {
	name  string
	arity int
	zip   func([]Observable[int]) Observable[[]int]
} {
	return []struct {
		name  string
		arity int
		zip   func([]Observable[int]) Observable[[]int]
	}{
		{"Zip", 2, func(s []Observable[int]) Observable[[]int] { return Zip(s...) }},
		{"ZipAll", 2, func(s []Observable[int]) Observable[[]int] { return ZipAll[int]()(Just(s...)) }},
		{"Zip2", 2, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple2[int, int]) []int { return []int{v.A, v.B} })(Zip2(s[0], s[1]))
		}},
		{"Zip3", 3, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple3[int, int, int]) []int { return []int{v.A, v.B, v.C} })(Zip3(s[0], s[1], s[2]))
		}},
		{"Zip4", 4, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple4[int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D} })(Zip4(s[0], s[1], s[2], s[3]))
		}},
		{"Zip5", 5, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple5[int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E} })(Zip5(s[0], s[1], s[2], s[3], s[4]))
		}},
		{"Zip6", 6, func(s []Observable[int]) Observable[[]int] {
			return Map(func(v lo.Tuple6[int, int, int, int, int, int]) []int { return []int{v.A, v.B, v.C, v.D, v.E, v.F} })(Zip6(s[0], s[1], s[2], s[3], s[4], s[5]))
		}},
	}
}

func TestOperatorCombiningZipCompletedSource(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		for short := 0; short < variant.arity; short++ {
			short := short
			t.Run(fmt.Sprintf("%s/source%d", variant.name, short), func(t *testing.T) {
				t.Parallel()
				testWithTimeout(t, 5*time.Second)
				is := assert.New(t)

				type contextKey struct{}
				subscriberCtx := context.WithValue(context.Background(), contextKey{}, "subscriber")
				emissionCtx := context.WithValue(subscriberCtx, contextKey{}, "last value")
				sources := make([]Observable[int], variant.arity)
				emitters := make([]Observer[int], variant.arity)
				teardowns := make([]int, variant.arity)
				for i := range sources {
					i := i
					sources[i] = NewObservableWithContext(func(ctx context.Context, destination Observer[int]) Teardown {
						is.Equal(subscriberCtx, ctx)
						emitters[i] = destination
						if i == short {
							destination.NextWithContext(ctx, 10+i)
							destination.NextWithContext(ctx, 20+i)
							destination.CompleteWithContext(ctx)
						}
						return func() { teardowns[i]++ }
					})
				}

				values := [][]int{}
				completions := 0
				sub := variant.zip(sources).SubscribeWithContext(subscriberCtx, NewObserverWithContext(
					func(ctx context.Context, value []int) {
						is.Equal(emissionCtx, ctx)
						values = append(values, value)
					},
					func(_ context.Context, err error) { is.NoError(err) },
					func(ctx context.Context) {
						is.Equal(emissionCtx, ctx)
						completions++
					},
				))
				defer sub.Unsubscribe()

				// Subscribe has returned, so completion now runs Zip's teardown.
				// The completed source must retain its second value until it is paired.
				want := make([][]int, 2)
				for round := 1; round <= 2; round++ {
					want[round-1] = make([]int, variant.arity)
					for i := range sources {
						want[round-1][i] = round*10 + i
						if i != short {
							emitters[i].NextWithContext(emissionCtx, round*10+i)
						}
					}
					is.Len(values, round)
					is.Equal(round-1, completions)
				}

				is.Equal(want, values)
				is.True(sub.IsClosed())
				for _, count := range teardowns {
					is.Equal(1, count)
				}
			})
		}
	}
}

func TestOperatorCombiningZipFutureCompletion(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()
			testWithTimeout(t, 5*time.Second)
			is := assert.New(t)
			release := make(chan struct{})
			sources := make([]Observable[int], variant.arity)
			for i := range sources {
				i := i
				sources[i] = Future(func() (int, error) {
					<-release
					return i, nil
				})
			}
			values := [][]int{}
			completions := 0
			sub := variant.zip(sources).Subscribe(NewObserver(
				func(value []int) { values = append(values, value) },
				func(err error) { is.NoError(err) },
				func() { completions++ },
			))
			defer sub.Unsubscribe()
			close(release)
			sub.Wait()

			want := make([]int, variant.arity)
			for i := range want {
				want[i] = i
			}
			is.Equal([][]int{want}, values)
			is.Equal(1, completions)
		})
	}
}

func TestOperatorCombiningZipUnsubscribeFromNext(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		t.Run(variant.name, func(t *testing.T) {
			t.Parallel()
			testWithTimeout(t, 5*time.Second)
			is := assert.New(t)
			sources := make([]Observable[int], variant.arity)
			emitters := make([]Observer[int], variant.arity)
			teardowns := 0
			for i := range sources {
				i := i
				sources[i] = NewObservable(func(destination Observer[int]) Teardown {
					emitters[i] = destination
					return func() { teardowns++ }
				})
			}
			values := 0
			completions := 0
			var sub Subscription
			sub = variant.zip(sources).Subscribe(NewObserver(
				func(_ []int) {
					values++
					sub.Unsubscribe()
				},
				func(err error) { is.NoError(err) },
				func() { completions++ },
			))
			defer sub.Unsubscribe()
			for _, emit := range emitters {
				emit.Next(1)
			}
			is.Equal(1, values)
			is.Zero(completions)
			is.True(sub.IsClosed())
			is.Equal(variant.arity, teardowns)
		})
	}
}

func TestOperatorCombiningZipTerminalCleanup(t *testing.T) {
	t.Parallel()

	for _, variant := range zipCompletionVariants() {
		variant := variant
		for _, terminal := range []string{"complete", "error", "cancel"} {
			terminal := terminal
			t.Run(variant.name+"/"+terminal, func(t *testing.T) {
				t.Parallel()
				testWithTimeout(t, 5*time.Second)
				is := assert.New(t)
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				teardowns := make(chan struct{}, variant.arity)
				sources := make([]Observable[int], variant.arity)
				var emit Observer[int]
				for i := range sources {
					sources[i] = NewObservable(func(destination Observer[int]) Teardown {
						emit = destination
						return func() { teardowns <- struct{}{} }
					})
				}
				if terminal == "cancel" {
					sources[len(sources)-1] = ThrowOnContextCancel[int]()(sources[len(sources)-1])
				}
				values := 0
				completions := 0
				var gotErr error
				sub := variant.zip(sources).SubscribeWithContext(ctx, NewObserver(
					func(_ []int) { values++ },
					func(err error) { gotErr = err },
					func() { completions++ },
				))
				defer sub.Unsubscribe()
				switch terminal {
				case "complete":
					emit.Complete()
				case "error":
					emit.Error(assert.AnError)
				case "cancel":
					cancel()
				}
				for range sources {
					<-teardowns
				}
				is.Zero(values)
				is.True(sub.IsClosed())
				switch terminal {
				case "complete":
					is.Equal(1, completions)
					is.NoError(gotErr)
				case "error":
					is.Zero(completions)
					is.ErrorIs(gotErr, assert.AnError)
				case "cancel":
					is.Zero(completions)
					is.ErrorIs(gotErr, context.Canceled)
				}
			})
		}
	}
}
