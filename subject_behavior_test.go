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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestBehaviorSubject_internalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(1).(*behaviorSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Equal(1, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send value
	subject.Next(21)
	is.Equal(KindNext, subject.status)
	is.Equal(21, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// completed state
	subject.Complete()
	is.Equal(KindComplete, subject.status)
	is.Equal(21, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(42)
	is.Equal(KindComplete, subject.status)
	is.Equal(21, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestBehaviorSubject_internalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(1).(*behaviorSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Equal(1, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send value
	subject.Next(21)
	is.Equal(KindNext, subject.status)
	is.Equal(21, subject.last.B)
	is.NoError(subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// trigger error
	subject.Error(assert.AnError)
	is.Equal(KindError, subject.status)
	is.Equal(21, subject.last.B)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(42)
	is.Equal(KindError, subject.status)
	is.Equal(21, subject.last.B)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestBehaviorSubject_internalSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

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

func TestBehaviorSubject_hasObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

	// default state
	is.True(ok)
	is.False(subject.HasObserver())
	subscription := subject.Subscribe(OnNext(func(value int) {}))
	is.True(subject.HasObserver())
	subscription.Unsubscribe()
	is.False(subject.HasObserver())
}

func TestBehaviorSubject_hasThrown(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
	subject2, ok2 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
	subject3, ok3 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

	// default state
	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	subject1.Next(42)
	subject2.Error(assert.AnError)
	subject3.Complete()
	is.False(subject1.HasThrown())
	is.True(subject2.HasThrown())
	is.False(subject3.HasThrown())
}

func TestBehaviorSubject_isComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
	subject2, ok2 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
	subject3, ok3 := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

	// default state
	is.True(ok1)
	is.True(ok2)
	is.True(ok3)
	subject1.Next(42)
	subject2.Error(assert.AnError)
	subject3.Complete()
	is.False(subject1.IsCompleted())
	is.False(subject2.IsCompleted())
	is.True(subject3.IsCompleted())
}

func TestBehaviorSubject_singleSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
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

func TestBehaviorSubject_multipleSubscription(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
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

func TestBehaviorSubject_subscriptionCanceledTwice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])
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

func TestBehaviorSubject_next(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

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
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	time.Sleep(10 * time.Millisecond)
	subject.Next(21)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(84), atomic.LoadInt64(&counter1))
	is.Equal(int64(84), atomic.LoadInt64(&counter2))
	is.Equal(int64(84), atomic.LoadInt64(&counter3))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
}

func TestBehaviorSubject_error(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

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
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Error(assert.AnError)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

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

func TestBehaviorSubject_reentrantCallFromObserver(t *testing.T) {
	// @TODO: Known bug. NextWithContext, and SubscribeWithContext when it replays
	// the last value, hold s.mu while they run observer callbacks. An observer
	// calling back into the subject (here IsClosed) deadlocks. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			_ = subject.IsClosed() // takes s.mu again
		}))
		subject.Next(1)
	}), "deadlocked on re-entrant IsClosed")
}

func TestBehaviorSubject_complete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewBehaviorSubject(42).(*behaviorSubjectImpl[int])

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
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// trigger error
	time.Sleep(10 * time.Millisecond)
	subject.Complete()
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// send a new message
	time.Sleep(10 * time.Millisecond)
	subject.Next(42)
	time.Sleep(10 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

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

// behaviorFaultyObserver delegates to a no-op observer and can panic on demand.
// Observers built with NewObserver recover callback panics, so they would never reach the subject.
type behaviorFaultyObserver struct {
	Observer[int]

	panicOnNext     bool // panics on value 1 only: the replay of the initial value must not panic
	panicOnTerminal bool
}

func (o behaviorFaultyObserver) Next(v int) {
	if o.panicOnNext && v == 1 {
		panic("boom")
	}

	o.Observer.Next(v)
}

func (o behaviorFaultyObserver) NextWithContext(ctx context.Context, v int) {
	if o.panicOnNext && v == 1 {
		panic("boom")
	}

	o.Observer.NextWithContext(ctx, v)
}

func (o behaviorFaultyObserver) Error(err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Error(err)
}

func (o behaviorFaultyObserver) ErrorWithContext(ctx context.Context, err error) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.ErrorWithContext(ctx, err)
}

func (o behaviorFaultyObserver) Complete() {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.Complete()
}

func (o behaviorFaultyObserver) CompleteWithContext(ctx context.Context) {
	if o.panicOnTerminal {
		panic("boom")
	}

	o.Observer.CompleteWithContext(ctx)
}

func newBehaviorFaultyObserver(panicOnNext, panicOnTerminal bool) Observer[int] {
	return behaviorFaultyObserver{
		Observer:        OnNext(func(int) {}),
		panicOnNext:     panicOnNext,
		panicOnTerminal: panicOnTerminal,
	}
}

func TestBehaviorSubject_reentrantNextFromObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Next(2) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Next")
}

func TestBehaviorSubject_reentrantSubscribeFromObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Subscribe(OnNext(func(int) {})) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Subscribe")
}

func TestBehaviorSubject_reentrantCompleteFromObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Complete() // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Complete")
}

func TestBehaviorSubject_reentrantErrorFromObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(v int) {
			if v == 1 {
				subject.Error(assert.AnError) // takes s.mu again
			}
		}))
		subject.Next(1)
	}), "Next deadlocked on re-entrant Error")
}

func TestBehaviorSubject_reentrantNextDuringReplay(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)

	// Subscribe replays the current value to the new observer.
	is.True(returnsWithin(500*time.Millisecond, func() {
		subject.Subscribe(OnNext(func(int) {
			_ = subject.IsClosed() // takes s.mu again
		}))
	}), "Subscribe deadlocked on re-entrant IsClosed while replaying")
}

func TestBehaviorSubject_reentrantCallFromCompleteObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)
	subject.Subscribe(OnComplete[int](func() {
		_ = subject.IsCompleted() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Complete() }), "Complete deadlocked on re-entrant IsCompleted")
}

func TestBehaviorSubject_reentrantCallFromErrorObserver(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)
	subject.Subscribe(OnError[int](func(error) {
		_ = subject.HasThrown() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Error(assert.AnError) }), "Error deadlocked on re-entrant HasThrown")
}

func TestBehaviorSubject_slowObserverBlocksQueries(t *testing.T) {
	// @TODO: Known bug. The behavior subject holds s.mu while it runs observer callbacks.
	// An observer calling back into the subject, or a slow observer, blocks it. Unskip once the
	// subject stops holding its lock across user code.
	t.Skip("known deadlock: behavior subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	release := make(chan struct{})
	entered := make(chan struct{})

	// Registered first so the blocked observer is released even when an assertion fails.
	t.Cleanup(func() { close(release) })

	subject := NewBehaviorSubject(0)
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

func TestBehaviorSubject_panickingObserverOnNextKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the behavior subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)
	subject.Subscribe(newBehaviorFaultyObserver(true, false))

	is.Panics(func() { subject.Next(1) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Next")
}

func TestBehaviorSubject_panickingObserverOnCompleteKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the behavior subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)
	subject.Subscribe(newBehaviorFaultyObserver(false, true))

	is.Panics(func() { subject.Complete() })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Complete")
}

func TestBehaviorSubject_panickingObserverOnErrorKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext, ErrorWithContext and CompleteWithContext release s.mu with a
	// plain Unlock, so a panic raised by an observer leaves the subject locked forever.
	// Unskip once the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the behavior subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewBehaviorSubject(0)
	subject.Subscribe(newBehaviorFaultyObserver(false, true))

	is.Panics(func() { subject.Error(assert.AnError) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic on Error")
}
