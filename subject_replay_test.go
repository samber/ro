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

func TestReplaySubject_internalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// completed state
	subject.Complete()
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(84)
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestReplaySubject_internalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// trigger error
	subject.Error(assert.AnError)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(42)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal([]int{21, 42}, t2ToSliceB(subject.values))
	is.Equal(10, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestReplaySubject_internalSubscription(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](42).(*replaySubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(0, subject.CountObservers())

	// subscribe
	sub1 := subject.Subscribe(NoopObserver[int]())
	is.Equal(uint32(1), subject.observerIndex)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(1, subject.CountObservers())

	// unsubscribe
	sub1.Unsubscribe()
	is.Equal(uint32(1), subject.observerIndex)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(0, subject.CountObservers())

	// resubscribe before completion
	sub2 := subject.Subscribe(NoopObserver[int]())
	is.Equal(uint32(2), subject.observerIndex)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(1, subject.CountObservers())

	// completed state
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(uint32(2), subject.observerIndex)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(0, subject.CountObservers())

	// no change
	sub3 := subject.Subscribe(NoopObserver[int]())
	is.Equal(uint32(2), subject.observerIndex)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(0, subject.CountObservers())

	sub2.Unsubscribe()
	sub3.Unsubscribe()
}

func TestReplaySubject_internalOverflow(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](2).(*replaySubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal([]int{}, t2ToSliceB(subject.values))
	is.Equal(2, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send values
	subject.Next(21)
	subject.Next(42)
	subject.Next(84)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	// the circular buffer overwrites the oldest value in place: 84 replaced 21
	is.Equal([]int{84, 42}, t2ToSliceB(subject.values))
	is.Equal(1, subject.head)
	is.Equal(2, subject.bufferSize)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// new subscribers replay values oldest-first
	values := []int{}
	subject.Subscribe(OnNext(func(value int) {
		values = append(values, value)
	})).Unsubscribe()
	is.Equal([]int{42, 84}, values)
}

func TestReplaySubject_hasObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

	is.True(ok)

	// default state
	is.False(subject.HasObserver())
	subscription := subject.Subscribe(OnNext(func(value int) {}))
	is.True(subject.HasObserver())
	subscription.Unsubscribe()
	is.False(subject.HasObserver())
}

func TestReplaySubject_hasThrown(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	subject2, ok2 := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	subject3, ok3 := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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

func TestReplaySubject_isComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	subject2, ok2 := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	subject3, ok3 := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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

func TestReplaySubject_singleSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe single
	subscription1 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(uint32(1), subject.observerIndex)

	// unsubscribe single
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(1), subject.observerIndex)
}

func TestReplaySubject_multipleSubscription(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe first
	subscription1 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(uint32(1), subject.observerIndex)

	// subscribe second
	subscription2 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.Equal(2, syncMapLength(&subject.observers))
	is.Equal(uint32(2), subject.observerIndex)

	// unsubscribe first
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(uint32(2), subject.observerIndex)

	// subscribe third
	subscription3 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.Equal(2, syncMapLength(&subject.observers))
	is.Equal(uint32(3), subject.observerIndex)

	// unsubscribe all
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(3), subject.observerIndex)
}

func TestReplaySubject_subscriptionCanceledTwice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])
	observer := OnNext(func(value int) {})

	is.True(ok)

	// subscribe single
	subscription1 := subject.Subscribe(observer)
	subscription2 := subject.Subscribe(observer)
	is.Equal(KindNext, subject.status)
	is.Equal(2, syncMapLength(&subject.observers))
	is.Equal(uint32(2), subject.observerIndex)

	// unsubscribe single
	subscription1.Unsubscribe()
	subscription1.Unsubscribe()
	is.Equal(KindNext, subject.status)
	is.Equal(1, syncMapLength(&subject.observers))
	is.Equal(uint32(2), subject.observerIndex)

	// clean before test exit
	subscription2.Unsubscribe()
}

func TestReplaySubject_next(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
}

func TestReplaySubject_error(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Error(assert.AnError)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// resubscribe
	subscription4 := subject.Subscribe(incOnNext(&counter4))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestReplaySubject_complete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	// resubscribe
	subscription4 := subject.Subscribe(incOnNext(&counter4))

	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestReplaySubject_reentrantCallFromObserver(t *testing.T) {
	// @TODO: Known bug. NextWithContext holds s.mu while it runs observer
	// callbacks, so an observer calling back into the subject (here IsClosed)
	// deadlocks. Unskip once the subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](1)
	subject.Subscribe(OnNext(func(int) {
		_ = subject.IsClosed() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Next(1) }), "Next deadlocked on re-entrant IsClosed")
}

func TestReplaySubject_replay(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](10).(*replaySubjectImpl[int])

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
	is.Equal(int64(84), atomic.LoadInt64(&counter2))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(84), atomic.LoadInt64(&counter2))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(84), atomic.LoadInt64(&counter2))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
}

func TestReplaySubject_implBufferOverflow(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewReplaySubject[int](2).(*replaySubjectImpl[int])

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
	is.Equal(int64(42), atomic.LoadInt64(&counter2))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(42), atomic.LoadInt64(&counter2))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(42), atomic.LoadInt64(&counter2))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
}

func TestReplaySubject_invalidBufferSize(t *testing.T) {
	t.Parallel()

	t.Run("negative size panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrReplaySubjectWrongBufferSize, func() { NewReplaySubject[int](-2) })
	})

	t.Run("min int size panics", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.PanicsWithValue(ErrReplaySubjectWrongBufferSize, func() { NewReplaySubject[int](math.MinInt) })
	})

	t.Run("unlimited size does not panic", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.NotPanics(func() { NewReplaySubject[int](ReplaySubjectUnlimitedBufferSize) })
	})

	t.Run("zero size does not panic", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		is.NotPanics(func() { NewReplaySubject[int](0) })
	})
}

// replayFaultyObserver delegates to a no-op observer and can panic on demand.
// Observers built with NewObserver recover callback panics, so they would never reach the subject.
type replayFaultyObserver struct {
	Observer[int]

	panicOnNext     bool // panics on value 1 only: the replay of the initial value must not panic
	panicOnTerminal bool
}

func (o replayFaultyObserver) Next(v int) {
	if o.panicOnNext && v == 1 {
		panic("boom")
	}

	o.Observer.Next(v)
}

func (o replayFaultyObserver) NextWithContext(ctx context.Context, v int) {
	if o.panicOnNext && v == 1 {
		panic("boom")
	}

	o.Observer.NextWithContext(ctx, v)
}

func (o replayFaultyObserver) Error(err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Error(err)
}

func (o replayFaultyObserver) ErrorWithContext(ctx context.Context, err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.ErrorWithContext(ctx, err)
}

func (o replayFaultyObserver) Complete() {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Complete()
}

func (o replayFaultyObserver) CompleteWithContext(ctx context.Context) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.CompleteWithContext(ctx)
}

func newReplayFaultyObserver(panicOnNext, panicOnTerminal bool) Observer[int] {
	return replayFaultyObserver{
		Observer:        OnNext(func(int) {}),
		panicOnNext:     panicOnNext,
		panicOnTerminal: panicOnTerminal,
	}
}

func TestReplaySubject_reentrantNextFromObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Next(2) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Next")
}

func TestReplaySubject_reentrantSubscribeFromObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Subscribe(OnNext(func(int) {})) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Subscribe")
}

func TestReplaySubject_reentrantCompleteFromObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Complete() // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Complete")
}

func TestReplaySubject_reentrantErrorFromObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Error(assert.AnError) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Error")
}

func TestReplaySubject_reentrantNextDuringReplay(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Next(0) // buffered, replayed to the next subscriber

	// Subscribe replays the buffered values to the new observer.
	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			_ = subject.IsClosed() // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant IsClosed while replaying")
}

func TestReplaySubject_reentrantCallFromCompleteObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Subscribe(OnComplete[int](func() {
		_ = subject.IsCompleted() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Complete() }), "Complete deadlocked on re-entrant IsCompleted")
}

func TestReplaySubject_reentrantCallFromErrorObserver(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Subscribe(OnError[int](func(error) {
		_ = subject.HasThrown() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Error(assert.AnError) }), "Error deadlocked on re-entrant HasThrown")
}

func TestReplaySubject_slowObserverBlocksQueries(t *testing.T) {
	// @TODO: Known bug. The replay subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: replay subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	release := make(chan struct{})
	entered := make(chan struct{})

	// Registered first so the blocked observer is released even when an assertion fails.
	t.Cleanup(func() { close(release) })

	subject := NewReplaySubject[int](10)
	subject.Subscribe(OnNext(func(v int) {
		if v == 1 {
			close(entered)
			<-release
		}
	}))

	go subject.Next(1)

	select {
	case <-entered:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("observer never called")
	}

	// A slow observer must not freeze unrelated readers of the subject.
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "IsClosed blocked by a slow observer")
	is.True(returnsWithin(500*time.Millisecond, func() { subject.Subscribe(OnNext(func(int) {})) }), "Subscribe blocked by a slow observer")
}

func TestReplaySubject_panickingObserverOnNextKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the replay subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Subscribe(newReplayFaultyObserver(true, false))

	is.Panics(func() { subject.Next(1) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Next")
}

func TestReplaySubject_panickingObserverOnCompleteKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the replay subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Subscribe(newReplayFaultyObserver(false, true))

	is.Panics(func() { subject.Complete() })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Complete")
}

func TestReplaySubject_panickingObserverOnErrorKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the replay subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewReplaySubject[int](10)
	subject.Subscribe(newReplayFaultyObserver(false, true))

	is.Panics(func() { subject.Error(assert.AnError) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Error")
}
