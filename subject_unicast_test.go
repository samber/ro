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
	"math"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestUnicastSubject_internalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// completed state
	subject.Complete()
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// no change
	subject.Next(84)
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)
}

func TestUnicastSubject_internalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// trigger error
	subject.Error(assert.AnError)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)

	// no change
	subject.Next(42)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Nil(subject.observer)
}

func TestUnicastSubject_internalSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](42).(*unicastSubjectImpl[int])

	is.True(ok)

	// default state
	is.Nil(subject.observer)
	is.Equal(0, subject.CountObservers())

	// subscribe
	sub1 := subject.Subscribe(NoopObserver[int]())
	is.NotNil(subject.observer)
	is.Equal(1, subject.CountObservers())

	// unsubscribe
	sub1.Unsubscribe()
	is.Nil(subject.observer)
	is.Equal(0, subject.CountObservers())

	// resubscribe before completion
	sub2 := subject.Subscribe(NoopObserver[int]())
	is.NotNil(subject.observer)
	is.Equal(1, subject.CountObservers())

	// completed state
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Nil(subject.observer)
	is.Equal(0, subject.CountObservers())

	// no change
	sub3 := subject.Subscribe(NoopObserver[int]())
	is.Nil(subject.observer)
	is.Equal(0, subject.CountObservers())

	sub2.Unsubscribe()
	sub3.Unsubscribe()
}

func TestUnicastSubject_internalOverflow(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](2).(*unicastSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(2, subject.bufferSize)
	is.Nil(subject.observer)

	// send values
	subject.Next(21)
	subject.Next(42)
	subject.Next(84)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{42, 84}, t2ToSliceB(subject.values))
	is.Equal(2, subject.bufferSize)
	is.Nil(subject.observer)
}

func TestUnicastSubject_hasObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	// default state
	is.False(subject.HasObserver())
	subscription := subject.Subscribe(OnNext(func(value int) {}))
	is.True(subject.HasObserver())
	subscription.Unsubscribe()
	is.False(subject.HasObserver())
}

func TestUnicastSubject_hasThrown(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	subject2, ok2 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	subject3, ok3 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)

	// default state
	subject1.Next(42)
	subject2.Error(assert.AnError)
	subject3.Complete()
	is.False(subject1.HasThrown())
	is.True(subject2.HasThrown())
	is.False(subject3.HasThrown())
}

func TestUnicastSubject_isComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	subject2, ok2 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	subject3, ok3 := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok1)
	is.True(ok2)
	is.True(ok3)

	// default state
	subject1.Next(42)
	subject2.Error(assert.AnError)
	subject3.Complete()
	is.False(subject1.IsCompleted())
	is.False(subject2.IsCompleted())
	is.True(subject3.IsCompleted())
}

func TestUnicastSubject_singleSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe single
	subscription1 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.NotNil(subject.observer)

	// unsubscribe single
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Nil(subject.observer)
}

func TestUnicastSubject_multipleSubscription(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe first
	subscription1 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.NotNil(subject.observer)

	// subscribe second
	subscription2 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.NotNil(subject.observer)

	// unsubscribe first
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Nil(subject.observer)

	// subscribe third
	subscription3 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.NotNil(subject.observer)

	// unsubscribe all
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Nil(subject.observer)
}

func TestUnicastSubject_subscriptionCanceledTwice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe single
	subscription1 := subject.Subscribe(observer)
	subscription2 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.NotNil(subject.observer)

	// unsubscribe single
	subscription1.Unsubscribe()
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Nil(subject.observer)

	// clean before test exit
	subscription2.Unsubscribe()
}

func TestUnicastSubject_next(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	var counter1 int64
	var counter2 int64
	var counter3 int64

	incOnNext := func(counter *int64) Observer[int] {
		return OnNext(func(value int) { atomic.AddInt64(counter, int64(value)) })
	}

	// subscribe 3 times
	subscription1 := subject.Subscribe(incOnNext(&counter1))
	subscription2 := subject.Subscribe(incOnNext(&counter2))
	subscription3 := subject.Subscribe(incOnNext(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
}

func TestUnicastSubject_error(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	var counter1 int64
	var counter2 int64
	var counter3 int64
	var counter4 int64

	incOnNext := func(counter *int64) Observer[int] {
		return OnNext(func(value int) { atomic.AddInt64(counter, int64(value)) })
	}

	// subscribe 3 times
	subscription1 := subject.Subscribe(incOnNext(&counter1))
	subscription2 := subject.Subscribe(incOnNext(&counter2))
	subscription3 := subject.Subscribe(incOnNext(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Error(assert.AnError)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// resubscribe
	subscription4 := subject.Subscribe(incOnNext(&counter4))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(0), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestUnicastSubject_complete(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	var counter1 int64
	var counter2 int64
	var counter3 int64
	var counter4 int64

	incOnNext := func(counter *int64) Observer[int] {
		return OnNext(func(value int) { atomic.AddInt64(counter, int64(value)) })
	}

	// subscribe 3 times
	subscription1 := subject.Subscribe(incOnNext(&counter1))
	subscription2 := subject.Subscribe(incOnNext(&counter2))
	subscription3 := subject.Subscribe(incOnNext(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))
	is.Equal(int64(0), atomic.LoadInt64(&counter3))

	// resubscribe
	subscription4 := subject.Subscribe(incOnNext(&counter4))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(0), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestUnicastSubject_reentrantCallFromObserver(t *testing.T) {
	// @TODO: Known bug. Next, Error and Complete deliver outside s.mu, but
	// SubscribeWithContext replays the buffered values while holding it. An
	// observer calling back into the subject (here HasObserver) during that replay
	// deadlocks. Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](1)
	subject.Next(1) // buffered, no observer yet

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			_ = subject.HasObserver() // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant HasObserver while replaying")
}

func TestUnicastSubject_replay(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](10).(*unicastSubjectImpl[int])

	is.True(ok)

	var counter1 int64
	var counter2 int64

	incOnNext := func(counter *int64) Observer[int] {
		return OnNext(func(value int) { atomic.AddInt64(counter, int64(value)) })
	}

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	subject.Next(21)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)

	subscription1 := subject.Subscribe(incOnNext(&counter1))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	subscription2 := subject.Subscribe(incOnNext(&counter2))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
}

func TestUnicastSubject_implBufferOverflow(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewUnicastSubject[int](2).(*unicastSubjectImpl[int])

	is.True(ok)

	var counter1 int64
	var counter2 int64

	incOnNext := func(counter *int64) Observer[int] {
		return OnNext(func(value int) { atomic.AddInt64(counter, int64(value)) })
	}

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	subject.Next(21)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)

	subscription1 := subject.Subscribe(incOnNext(&counter1))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(42), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	subscription2 := subject.Subscribe(incOnNext(&counter2))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(0), atomic.LoadInt64(&counter2))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
}

func TestUnicastSubject_invalidBufferSize(t *testing.T) {
	t.Parallel()

	t.Run("negative size panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrUnicastSubjectWrongBufferSize, func() { NewUnicastSubject[int](-2) })
	})

	t.Run("min int size panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrUnicastSubjectWrongBufferSize, func() { NewUnicastSubject[int](math.MinInt) })
	})

	t.Run("unlimited size does not panic", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.NotPanics(func() { NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize) })
	})

	t.Run("zero size does not panic", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.NotPanics(func() { NewUnicastSubject[int](0) })
	})
}

// unicastFaultyObserver delegates to a no-op observer and can panic on demand.
// Observers built with NewObserver recover callback panics, so they would never reach the subject.
type unicastFaultyObserver struct {
	Observer[int]

	panicOnNext     bool
	panicOnTerminal bool
}

func (o unicastFaultyObserver) Next(v int) {
	if o.panicOnNext {
		panic("boom")
	}

	o.Observer.Next(v)
}

func (o unicastFaultyObserver) NextWithContext(ctx context.Context, v int) {
	if o.panicOnNext {
		panic("boom")
	}

	o.Observer.NextWithContext(ctx, v)
}

func (o unicastFaultyObserver) Error(err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Error(err)
}

func (o unicastFaultyObserver) ErrorWithContext(ctx context.Context, err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.ErrorWithContext(ctx, err)
}

func (o unicastFaultyObserver) Complete() {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Complete()
}

func (o unicastFaultyObserver) CompleteWithContext(ctx context.Context) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.CompleteWithContext(ctx)
}

func newUnicastFaultyObserver(panicOnNext, panicOnTerminal bool) Observer[int] {
	return unicastFaultyObserver{
		Observer:        OnNext(func(int) {}),
		panicOnNext:     panicOnNext,
		panicOnTerminal: panicOnTerminal,
	}
}

func TestUnicastSubject_reentrantNextDuringReplay(t *testing.T) {
	// @TODO: Known bug. SubscribeWithContext replays the buffered values while holding s.mu.
	// An observer calling back into the subject during that replay, or a slow replay, blocks it.
	// Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			subject.Next(2) // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant Next while replaying")
}

func TestUnicastSubject_reentrantSubscribeDuringReplay(t *testing.T) {
	// @TODO: Known bug. SubscribeWithContext replays the buffered values while holding s.mu.
	// An observer calling back into the subject during that replay, or a slow replay, blocks it.
	// Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			subject.Subscribe(OnNext(func(int) {})) // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant Subscribe while replaying")
}

func TestUnicastSubject_reentrantCompleteDuringReplay(t *testing.T) {
	// @TODO: Known bug. SubscribeWithContext replays the buffered values while holding s.mu.
	// An observer calling back into the subject during that replay, or a slow replay, blocks it.
	// Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			subject.Complete() // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant Complete while replaying")
}

func TestUnicastSubject_reentrantErrorDuringReplay(t *testing.T) {
	// @TODO: Known bug. SubscribeWithContext replays the buffered values while holding s.mu.
	// An observer calling back into the subject during that replay, or a slow replay, blocks it.
	// Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			subject.Error(assert.AnError) // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant Error while replaying")
}

func TestUnicastSubject_slowReplayBlocksQueries(t *testing.T) {
	// @TODO: Known bug. SubscribeWithContext replays the buffered values while holding s.mu.
	// An observer calling back into the subject during that replay, or a slow replay, blocks it.
	// Unskip once the replay runs outside the lock.
	t.Skip("known deadlock: unicast subject replays buffered values under its mutex")

	t.Parallel()
	is := assert.New(t)

	release := make(chan struct{})
	entered := make(chan struct{})

	// Registered first so the blocked observer is released even when an assertion fails.
	t.Cleanup(func() { close(release) })

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	go subject.Subscribe(OnNext(func(int) {
		close(entered)
		<-release
	}))

	select {
	case <-entered:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("observer never called")
	}

	// A slow replay must not freeze unrelated readers of the subject.
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "IsClosed blocked by a slow replay")
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.CountObservers() }), "CountObservers blocked by a slow replay")
}

func TestUnicastSubject_replayLeavesSubjectUsableAfterObserverPanic(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
	subject.Next(1) // buffered, no observer yet

	is.Panics(func() { subject.Subscribe(newUnicastFaultyObserver(true, false)) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic while replaying")
}

func TestUnicastSubject_reentrantCallFromLiveObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// Live notifications are delivered outside s.mu: every re-entrant query must return.
	tests := map[string]func(subject Subject[int]){
		"Subscribe":   func(subject Subject[int]) { subject.Subscribe(OnNext(func(int) {})) },
		"HasObserver": func(subject Subject[int]) { _ = subject.HasObserver() },
		"IsClosed":    func(subject Subject[int]) { _ = subject.IsClosed() },
	}

	for name, call := range tests {
		subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)

		var reentered atomic.Bool

		subject.Subscribe(OnNext(func(int) {
			if reentered.CompareAndSwap(false, true) {
				call(subject)
			}
		}))

		is.True(returnsWithin(500*time.Millisecond, func() { subject.Next(1) }), "Next deadlocked on re-entrant "+name)
	}
}

func TestUnicastSubject_reentrantUnsubscribeFromLiveObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)

	var subscription Subscription

	subscribed := make(chan struct{})

	subscription = subject.Subscribe(OnNext(func(int) {
		<-subscribed
		subscription.Unsubscribe() // teardown takes s.mu
	}))
	close(subscribed)

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Next(1) }), "Next deadlocked on Unsubscribe from the observer")
	is.False(subject.HasObserver())
}

func TestUnicastSubject_panickingObserverKeepsSubjectUsable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// A fresh subject per notification: the panic also leaves the subscriber wrapping the
	// observer locked, so it cannot take a second notification.
	tests := map[string]func(subject Subject[int]){
		"Next":     func(subject Subject[int]) { subject.Next(1) },
		"Error":    func(subject Subject[int]) { subject.Error(assert.AnError) },
		"Complete": func(subject Subject[int]) { subject.Complete() },
	}

	for name, notify := range tests {
		subject := NewUnicastSubject[int](UnicastSubjectUnlimitedBufferSize)
		subject.Subscribe(newUnicastFaultyObserver(true, true))

		is.Panics(func() { notify(subject) }, name)
		is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on "+name)
	}
}
