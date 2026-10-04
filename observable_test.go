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
	"strconv"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestObservable_lazy(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	// is := assert.New(t)

	// We check that the publisher is not started until we subscribe.
	_ = NewObservable(func(observer Observer[int]) Teardown {
		panic("never 1")
	})

	// We check that the publisher cancellation is not triggered until we subscribe.
	_ = NewObservable(func(observer Observer[int]) Teardown {
		return func() {
			panic("never 1")
		}
	})
}

func TestObservable_handleComplete(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(0)
			observer.Next(1)
			observer.Complete()
			observer.Next(2)

			return nil
		}),
	)
	is.Equal([]int{0, 1}, values)
	is.NoError(err)
}

func TestObservable_handleError(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(0)
			observer.Next(1)
			observer.Error(assert.AnError)
			observer.Next(2)

			return nil
		}),
	)
	is.Equal([]int{0, 1}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		NewObservable(func(observer Observer[int]) Teardown {
			observer.Next(0)
			observer.Next(1)
			observer.Error(nil)
			observer.Next(2)

			return nil
		}),
	)
	is.Equal([]int{0, 1}, values)
	is.NoError(err)
}

func TestObservable_handlePanic_string(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	done := false

	// We check that the panic is propagated to the observer as an error.
	obs := NewObservable(func(observer Observer[int]) Teardown {
		panic("hello world")
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.Fail("never")
			},
			func(err error) {
				is.Errorf(err, "unexpected error: hello world")

				done = true
			},
			func() {
				is.Fail("never")
			},
		),
	)

	sub.Unsubscribe()
	is.True(done)
}

func TestObservable_handlePanic_error(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	done := false

	// We check that the panic is propagated to the observer as an error.
	obs := NewObservable(func(observer Observer[int]) Teardown {
		panic(assert.AnError)
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.Fail("never")
			},
			func(err error) {
				is.Errorf(err, assert.AnError.Error())

				done = true
			},
			func() {
				is.Fail("never")
			},
		),
	)

	sub.Unsubscribe()
	is.True(done)
}

func TestObservable_nilTeardown(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	obs := NewObservable(func(observer Observer[int]) Teardown {
		observer.Next(42)
		return nil
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.Equal(42, v)
			},
			func(err error) {
				is.Fail("never")
			},
			func() {
				is.Fail("never")
			},
		),
	)

	sub.Unsubscribe()
}

func TestObservable_notNilTeardown(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	done := 0

	obs := NewObservable(func(observer Observer[int]) Teardown {
		observer.Next(42)

		return func() {
			done++
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.Equal(42, v)
			},
			func(err error) {
				is.Fail("never")
			},
			func() {
				is.Fail("never")
			},
		),
	)

	is.False(sub.IsClosed())
	is.Equal(0, done)
	sub.Unsubscribe()
	is.True(sub.IsClosed())
	is.Equal(1, done)
}

func TestObservable_panicTeardown(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	obs := NewObservable(func(observer Observer[int]) Teardown {
		observer.Next(42)

		return func() {
			panic(assert.AnError) // must crash at Unsuscribe()
		}
	})

	var sub Subscription

	// the panic is not propagated until Unsubscribe() is called
	is.NotPanics(
		func() {
			sub = obs.Subscribe(
				NewObserver(
					func(v int) {
						is.Equal(42, v)
					},
					func(err error) {
						is.Fail("never")
					},
					func() {
						is.Fail("never")
					},
				),
			)
		},
	)

	is.PanicsWithError(
		newUnsubscriptionError(assert.AnError).Error(),
		func() { sub.Unsubscribe() },
	)
}

func TestObservable_nonBlocking(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	counter := int32(0)

	// We check that calling Subscribe() is non blocking.
	// (the sleep at the beginning of publisher gives guarantee)
	obs := NewObservable(func(observer Observer[int]) Teardown {
		go func() {
			time.Sleep(50 * time.Millisecond)
			observer.Next(0)
			observer.Next(1)
			observer.Next(2)
			observer.Complete()
		}()

		return func() {
			time.Sleep(50 * time.Millisecond)
			atomic.AddInt32(&counter, 1)
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.EqualValues(v, atomic.LoadInt32(&counter))
				atomic.AddInt32(&counter, 1)
			},
			func(error) {
				panic("never")
			},
			func() {
				is.Equal(int32(3), atomic.LoadInt32(&counter))
				atomic.AddInt32(&counter, 1)
			},
		),
	)

	// ensure this is non-blocking
	is.False(sub.IsClosed())
	is.Equal(int32(0), atomic.LoadInt32(&counter))
	time.Sleep(200 * time.Millisecond)
	sub.Unsubscribe()
	is.Equal(int32(5), atomic.LoadInt32(&counter))
}

func TestObservable_blocking(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	counter := int32(0)

	// We check that calling Subscribe() is blocking.
	// (the sleep at the beginning of publisher gives guarantee)
	obs := NewObservable(func(observer Observer[int]) Teardown {
		time.Sleep(50 * time.Millisecond)
		observer.Next(0)
		observer.Next(1)
		observer.Next(2)
		observer.Complete()

		return func() {
			time.Sleep(50 * time.Millisecond)
			atomic.AddInt32(&counter, 1)
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v int) {
				is.EqualValues(v, atomic.LoadInt32(&counter))
				atomic.AddInt32(&counter, 1)
			},
			func(error) {
				panic("never")
			},
			func() {
				is.Equal(int32(3), atomic.LoadInt32(&counter))
				atomic.AddInt32(&counter, 1)
			},
		),
	)

	// ensure this is blocking
	is.True(sub.IsClosed())
	is.Equal(int32(5), atomic.LoadInt32(&counter))
	sub.Unsubscribe()
	is.Equal(int32(5), atomic.LoadInt32(&counter))
}

func TestObservable_blockOnDowntreamWork(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	result := ""
	mu := lo.Synchronize()

	obs := NewObservable(func(observer Observer[string]) Teardown {
		time.Sleep(50 * time.Millisecond)

		// Next() must be blocking, until the downstream work is done
		// Next() is released when every following operators have processed the message.
		observer.Next("a")
		mu.Do(func() { result += "b" })
		observer.Next("c")
		mu.Do(func() { result += "d" })
		observer.Next("e")
		mu.Do(func() { result += "f" })

		// Same for Complete()
		observer.Complete()
		mu.Do(func() { result += "h" })

		return func() {
			time.Sleep(50 * time.Millisecond)
			mu.Do(func() { result += "i" })
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v string) {
				time.Sleep(50 * time.Millisecond) // simulate long task
				mu.Do(func() { result += v })
			},
			func(error) {
				panic("never")
			},
			func() {
				time.Sleep(50 * time.Millisecond) // simulate long task
				mu.Do(func() { result += "g" })
			},
		),
	)
	defer sub.Unsubscribe()

	// ensure this is blocking
	is.True(sub.IsClosed())
	mu.Do(func() { is.Equal("abcdefghi", result) })
}

func TestObservable_blocking_cancelCompleted(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	result := ""
	mu := lo.Synchronize()

	obs := NewObservable(func(observer Observer[string]) Teardown {
		time.Sleep(50 * time.Millisecond)
		observer.Next("a")
		observer.Next("b")
		observer.Next("c")
		mu.Do(func() { result += "d" })
		observer.Complete()
		mu.Do(func() { result += "f" })

		return func() {
			time.Sleep(50 * time.Millisecond)
			mu.Do(func() { result += "g" })
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v string) {
				mu.Do(func() { result += v })
			},
			func(error) {
				panic("never")
			},
			func() {
				time.Sleep(50 * time.Millisecond)
				mu.Do(func() { result += "e" })
			},
		),
	)

	// the stream must be consumed
	is.True(sub.IsClosed())
	mu.Do(func() { result += "h" })
	mu.Do(func() { is.Equal("abcdefgh", result) })

	// noop
	sub.Unsubscribe()
	is.True(sub.IsClosed())
	mu.Do(func() { result += "i" })
	mu.Do(func() { is.Equal("abcdefghi", result) })
}

func TestObservable_nonBlocking_cancelCompleted(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	result := ""
	mu := lo.Synchronize()

	obs := NewObservable(func(observer Observer[string]) Teardown {
		go func() {
			time.Sleep(100 * time.Millisecond)
			observer.Next("a")
			observer.Next("b")
			observer.Next("c")
			mu.Do(func() { result += "d" })
			observer.Complete()
			mu.Do(func() { result += "g" })
		}()

		return func() {
			time.Sleep(50 * time.Millisecond)
			mu.Do(func() { result += "f" })
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v string) {
				mu.Do(func() { result += v })
			},
			func(error) {
				panic("never")
			},
			func() {
				time.Sleep(50 * time.Millisecond)
				mu.Do(func() { result += "e" })
			},
		),
	)

	// non-blocking
	is.False(sub.IsClosed())
	mu.Do(func() { is.Empty(result) })

	// ensure the stream is consumed
	time.Sleep(300 * time.Millisecond)
	is.True(sub.IsClosed())
	mu.Do(func() { result += "h" })
	mu.Do(func() { is.Equal("abcdefgh", result) })

	// blocking
	sub.Unsubscribe()
	is.True(sub.IsClosed())
	mu.Do(func() { result += "i" })
	mu.Do(func() { is.Equal("abcdefghi", result) })
}

func TestObservable_blocking_cancelActive(t *testing.T) { //nolint:paralleltest
	// N/A
}

func TestObservable_nonBlocking_cancelActive(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	result := ""
	mu := lo.Synchronize()

	obs := NewObservable(func(observer Observer[string]) Teardown {
		go func() {
			time.Sleep(100 * time.Millisecond)
			observer.Next("a")
			time.Sleep(100 * time.Millisecond)
			observer.Next("b")
			time.Sleep(100 * time.Millisecond)
			observer.Next("c")

			time.Sleep(100 * time.Millisecond)
			mu.Do(func() { result += "d" })
			observer.Complete()
			mu.Do(func() { result += "g" })
		}()

		return func() {
			time.Sleep(50 * time.Millisecond)
			mu.Do(func() { result += "f" })
		}
	})

	sub := obs.Subscribe(
		NewObserver(
			func(v string) {
				mu.Do(func() { result += v })
			},
			func(error) {
				panic("never")
			},
			func() {
				time.Sleep(50 * time.Millisecond)
				mu.Do(func() { result += "e" })
			},
		),
	)

	// non-blocking
	is.False(sub.IsClosed())
	mu.Do(func() { is.Empty(result) })

	// ensure the stream is consumed
	time.Sleep(150 * time.Millisecond)
	is.False(sub.IsClosed())
	mu.Do(func() { is.Equal("a", result) })

	// blocking
	sub.Unsubscribe()
	is.True(sub.IsClosed())
	mu.Do(func() { result += "h" })
	mu.Do(func() { is.Equal("afh", result) })
}

func TestObservable_chain(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	result := ""
	mu := lo.Synchronize()

	obs1 := NewObservable(func(observer Observer[string]) Teardown {
		observer.Next("0")
		observer.Next("1")
		observer.Next("2")
		observer.Complete()

		return nil
	})

	obs2 := NewObservable(func(observer Observer[string]) Teardown {
		sub := obs1.Subscribe(
			NewObserver(
				func(v string) {
					mu.Do(func() { result += v })
					observer.Next(v)
				},
				func(err error) {
					observer.Error(err)
				},
				func() {
					observer.Complete()
				},
			),
		)

		return sub.Unsubscribe
	})

	obs3 := NewObservable(func(observer Observer[string]) Teardown {
		sub := obs2.Subscribe(
			NewObserver(
				func(v string) {
					mu.Do(func() { result += v })
					observer.Next(v)
				},
				func(err error) {
					observer.Error(err)
				},
				func() {
					observer.Complete()
				},
			),
		)

		return sub.Unsubscribe
	})

	sub := obs3.Subscribe(
		NewObserver(
			func(v string) {
				mu.Do(func() { result += v })
			},
			func(err error) {
				panic("never")
			},
			func() {
				is.Equal("000111222", result)
			},
		),
	)

	sub.Unsubscribe()
	is.True(sub.IsClosed())
	is.Equal("000111222", result)
}

func TestObservable_contextPropagation(t *testing.T) {
	t.Parallel()
	// testWithTimeout(t, 10*time.Millisecond)
	is := assert.New(t)

	type ctxKey string

	ctxKey42 := ctxKey("42")
	ctxKey42Int := 42

	obs := Pipe2(
		Just(1, 2, 3),
		ContextWithValue[int](42, -42),
		MapWithContext(func(ctx context.Context, i int) (context.Context, int) {
			is.Equal("abcd", ctx.Value(ctxKey42))
			is.Equal(-42, ctx.Value(ctxKey42Int))

			return ctx, i*2 + ctx.Value(ctxKey42Int).(int) //nolint:errcheck,forcetypeassert
		}),
	)

	result, ctx, err := CollectWithContext(context.WithValue(context.Background(), ctxKey42, "abcd"), obs)
	is.Equal([]int{-40, -38, -36}, result)
	is.Equal("abcd", ctx.Value(ctxKey42))
	is.Equal(-42, ctx.Value(ctxKey42Int)) // because MapWithContext does not change context of Complete notification
	is.NoError(err)
}

func TestNewConnectableObservable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a := []int{}
	b := []string{}

	source := func(destination Observer[int]) Teardown {
		destination.Next(1)
		destination.Next(2)
		destination.Next(3)
		destination.Complete()

		return nil
	}

	connectable, ok := NewConnectableObservable(source).(*connectableObservableImpl[int])

	is.True(ok)

	is.True(connectable.config.ResetOnDisconnect)
	is.NotNil(connectable.config.Connector)
	is.NotNil(connectable.source)
	is.Nil(connectable.subscription)

	sub1 := connectable.Subscribe(OnNext(func(item int) {
		a = append(a, item)
	}))
	sub2 := connectable.Subscribe(OnNext(func(item int) {
		b = append(b, strconv.Itoa(item))
	}))

	is.Nil(connectable.subscription)
	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())

	sub := connectable.Connect()
	is.True(connectable.subscription.IsClosed())
	is.True(sub.IsClosed())
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())

	is.Equal([]int{1, 2, 3}, a)
	is.Equal([]string{"1", "2", "3"}, b)
}

func TestNewConnectableObservableWithConfig(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a := []int{}
	b := []string{}

	source := func(destination Observer[int]) Teardown {
		destination.Next(1)
		destination.Next(2)
		destination.Next(3)
		destination.Complete()

		return nil
	}

	config := ConnectableConfig[int]{
		Connector:         NewSubject[int],
		ResetOnDisconnect: true,
	}
	connectable, ok := NewConnectableObservableWithConfig(source, config).(*connectableObservableImpl[int])

	is.True(ok)

	is.True(connectable.config.ResetOnDisconnect)
	is.NotNil(connectable.config.Connector)
	is.NotNil(connectable.source)
	is.Nil(connectable.subscription)

	sub1 := connectable.Subscribe(OnNext(func(item int) {
		a = append(a, item)
	}))
	sub2 := connectable.Subscribe(OnNext(func(item int) {
		b = append(b, strconv.Itoa(item))
	}))

	is.Nil(connectable.subscription)
	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())

	sub := connectable.Connect()
	is.True(connectable.subscription.IsClosed())
	is.True(sub.IsClosed())
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())

	is.Equal([]int{1, 2, 3}, a)
	is.Equal([]string{"1", "2", "3"}, b)
}

func TestConnectable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a := []int{}
	b := []int{}
	c := []string{}

	source := TapOnNext(func(value int) {
		a = append(a, value*2)
	})(Of(1, 2, 3))

	connectable, ok := Connectable(source).(*connectableObservableImpl[int])

	is.True(ok)

	is.True(connectable.config.ResetOnDisconnect)
	is.NotNil(connectable.config.Connector)
	is.NotNil(connectable.source)
	is.Nil(connectable.subscription)

	sub1 := connectable.Subscribe(OnNext(func(item int) {
		b = append(b, item)
	}))
	sub2 := connectable.Subscribe(OnNext(func(item int) {
		c = append(c, strconv.Itoa(item))
	}))

	is.Nil(connectable.subscription)
	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())

	sub := connectable.Connect()
	is.True(connectable.subscription.IsClosed())
	is.True(sub.IsClosed())
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())

	is.Equal([]int{2, 4, 6}, a)
	is.Equal([]int{1, 2, 3}, b)
	is.Equal([]string{"1", "2", "3"}, c)
}

func TestConnectableWithConfig(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a := []int{}
	b := []int{}
	c := []string{}

	source := TapOnNext(func(value int) {
		a = append(a, value*2)
	})(Of(1, 2, 3))

	config := ConnectableConfig[int]{
		Connector:         NewSubject[int],
		ResetOnDisconnect: true,
	}
	connectable, ok := ConnectableWithConfig(source, config).(*connectableObservableImpl[int])

	is.True(ok)

	is.True(connectable.config.ResetOnDisconnect)
	is.NotNil(connectable.config.Connector)
	is.NotNil(connectable.source)
	is.Nil(connectable.subscription)

	sub1 := connectable.Subscribe(OnNext(func(item int) {
		b = append(b, item)
	}))
	sub2 := connectable.Subscribe(OnNext(func(item int) {
		c = append(c, strconv.Itoa(item))
	}))

	is.Nil(connectable.subscription)
	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())

	sub := connectable.Connect()
	is.True(connectable.subscription.IsClosed())
	is.True(sub.IsClosed())
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())

	is.Equal([]int{2, 4, 6}, a)
	is.Equal([]int{1, 2, 3}, b)
	is.Equal([]string{"1", "2", "3"}, c)
}
