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
	"strconv"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorConnectableShare(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []string{}
	d := []string{}

	source := Pipe3(
		Just(1, 2, 3),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		Delay[int](10*time.Millisecond),
		Share[int](),
	)

	sub1 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			b = append(b, item*2)
		})
	}))
	sub2 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			c = append(c, strconv.Itoa(item))
		})
	}))
	sub3 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			d = append(d, strconv.Itoa(item))
		})
	}))

	mu.Do(func() {
		is.Equal([]int{1, 2, 3}, a)
		is.Equal([]int{}, b)
		is.Equal([]string{}, c)
		is.Equal([]string{}, d)
	})

	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())
	is.False(sub3.IsClosed())

	sub1.Unsubscribe()
	is.True(sub1.IsClosed())
	is.False(sub2.IsClosed())
	is.False(sub3.IsClosed())

	sub2.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
	is.False(sub3.IsClosed())

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2, 3}, a)
		is.Equal([]int{}, b)
		is.Equal([]string{}, c)
		is.Equal([]string{"1", "2", "3"}, d)
	})

	sub3.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
	is.True(sub3.IsClosed())
}

func TestOperatorConnectableShareWithConfig(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []string{}
	d := []string{}

	source := Pipe3(
		Just(1, 2, 3),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		Delay[int](10*time.Millisecond),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     false,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			b = append(b, item*2)
		})
	}))
	sub2 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			c = append(c, strconv.Itoa(item))
		})
	}))
	sub3 := source.Subscribe(OnNext(func(item int) {
		mu.Do(func() {
			d = append(d, strconv.Itoa(item))
		})
	}))

	mu.Do(func() {
		is.Equal([]int{1, 2, 3}, a)
		is.Equal([]int{}, b)
		is.Equal([]string{}, c)
		is.Equal([]string{}, d)
	})

	is.False(sub1.IsClosed())
	is.False(sub2.IsClosed())
	is.False(sub3.IsClosed())

	sub1.Unsubscribe()
	is.True(sub1.IsClosed())
	is.False(sub2.IsClosed())
	is.False(sub3.IsClosed())

	sub2.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
	is.False(sub3.IsClosed())

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2, 3}, a)
		is.Equal([]int{}, b)
		is.Equal([]string{}, c)
		is.Equal([]string{"1", "2", "3"}, d)
	})

	sub3.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
	is.True(sub3.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnError_false(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}

	var err1 error
	var err2 error

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Error(assert.AnError)
				destination.Next(3)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     false,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
				mu.Do(func() {
					err1 = err
				})
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
		is.EqualError(assert.AnError, err1.Error())
		is.NoError(err2)
	})
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
				mu.Do(func() {
					err2 = err
				})
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c) // has not been reset
		is.EqualError(assert.AnError, err1.Error())
		is.EqualError(assert.AnError, err2.Error())
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnError_true(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}

	var err1 error
	var err2 error

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Error(assert.AnError)
				destination.Next(3)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        true,
			ResetOnComplete:     false,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
				mu.Do(func() {
					err1 = err
				})
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
		is.EqualError(assert.AnError, err1.Error())
		is.NoError(err2)
	})
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
				mu.Do(func() {
					err2 = err
				})
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2, 1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{4, 8}, c) // has been reset
		is.EqualError(assert.AnError, err1.Error())
		is.EqualError(assert.AnError, err2.Error())
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnComplete_false(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}
	completedA := false
	completedB := false

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Complete()
				destination.Next(3)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     false,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
			},
			func() {
				mu.Do(func() {
					completedA = true
				})
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
		is.True(completedA)
		is.False(completedB)
	})
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
			},
			func() {
				mu.Do(func() {
					completedB = true
				})
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c) // has not been reset
		is.True(completedA)
		is.True(completedB)
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnComplete_true(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}
	completedA := false
	completedB := false

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Complete()
				destination.Next(3)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     true,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
			},
			func() {
				mu.Do(func() {
					completedA = true
				})
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
		is.True(completedA)
		is.False(completedB)
	})
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
			},
			func() {
				mu.Do(func() {
					completedB = true
				})
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2, 1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{4, 8}, c) // has been reset
		is.True(completedA)
		is.True(completedB)
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnRefcount_false(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     false,
			ResetOnRefCountZero: false,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
	})
	is.False(sub1.IsClosed())
	sub1.Unsubscribe()
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a) // has not been reset
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c) // has not been reset
	})
	is.False(sub2.IsClosed())
	sub2.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareWithConfig_resetOnRefcount_true(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 500*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int{}
	b := []int{}
	c := []int{}

	source := Pipe2(
		NewObservable(func(destination Observer[int]) Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
			}()

			return nil
		}),
		TapOnNext(func(value int) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareWithConfig(ShareConfig[int]{
			Connector:           defaultConnector[int],
			ResetOnError:        false,
			ResetOnComplete:     false,
			ResetOnRefCountZero: true,
		}),
	)

	sub1 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					b = append(b, item*2)
				})
			},
			func(err error) {
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2}, a)
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{}, c)
	})
	is.False(sub1.IsClosed())
	sub1.Unsubscribe()
	is.True(sub1.IsClosed())

	sub2 := source.Subscribe(
		NewObserver(
			func(item int) {
				mu.Do(func() {
					c = append(c, item*4)
				})
			},
			func(err error) {
			},
			func() {
			},
		),
	)

	time.Sleep(50 * time.Millisecond)
	mu.Do(func() {
		is.Equal([]int{1, 2, 1, 2}, a) // has been reset
		is.Equal([]int{2, 4}, b)
		is.Equal([]int{4, 8}, c) // has been reset
	})
	is.False(sub2.IsClosed())
	sub2.Unsubscribe()
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareReplay(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int64{}
	b := []int64{}
	c := []int64{}

	source := Pipe2(
		RangeWithInterval(0, 5, 50*time.Millisecond),
		TapOnNext(func(value int64) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareReplay[int64](10),
	)

	sub1 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				b = append(b, item*2)
			})
		}),
	)

	time.Sleep(125 * time.Millisecond)

	sub2 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				c = append(c, item*4)
			})
		}),
	)

	time.Sleep(200 * time.Millisecond)

	mu.Do(func() {
		is.Equal([]int64{0, 1, 2, 3, 4}, a)
		is.Equal([]int64{0, 2, 4, 6, 8}, b)
		is.Equal([]int64{0, 4, 8, 12, 16}, c)
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareReplay_smallBuffer(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int64{}
	b := []int64{}
	c := []int64{}

	source := Pipe2(
		RangeWithInterval(0, 5, 50*time.Millisecond),
		TapOnNext(func(value int64) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareReplay[int64](1),
	)

	sub1 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				b = append(b, item*2)
			})
		}),
	)

	time.Sleep(125 * time.Millisecond)

	sub2 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				c = append(c, item*4)
			})
		}),
	)

	time.Sleep(200 * time.Millisecond)

	mu.Do(func() {
		is.Equal([]int64{0, 1, 2, 3, 4}, a)
		is.Equal([]int64{0, 2, 4, 6, 8}, b)
		is.Equal([]int64{4, 8, 12, 16}, c)
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}

func TestOperatorConnectableShareReplayWithConfig(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	testWithTimeout(t, 1000*time.Millisecond)
	is := assert.New(t)

	mu := lo.Synchronize()
	a := []int64{}
	b := []int64{}
	c := []int64{}

	source := Pipe2(
		RangeWithInterval(0, 5, 50*time.Millisecond),
		TapOnNext(func(value int64) {
			mu.Do(func() {
				a = append(a, value)
			})
		}),
		ShareReplayWithConfig[int64](2, ShareReplayConfig{
			ResetOnRefCountZero: true,
		}),
	)

	sub1 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				b = append(b, item*2)
			})
		}),
	)

	time.Sleep(125 * time.Millisecond)

	sub2 := source.Subscribe(
		OnNext(func(item int64) {
			mu.Do(func() {
				c = append(c, item*4)
			})
		}),
	)

	time.Sleep(200 * time.Millisecond)

	mu.Do(func() {
		is.Equal([]int64{0, 1, 2, 3, 4}, a)
		is.Equal([]int64{0, 2, 4, 6, 8}, b)
		is.Equal([]int64{0, 4, 8, 12, 16}, c)
	})
	is.True(sub1.IsClosed())
	is.True(sub2.IsClosed())
}
