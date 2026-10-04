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
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestPublishSubject_internalOk(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// completed state
	subject.Complete()
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(84)
	is.Equal(KindComplete, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestPublishSubject_internalError(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

	is.True(ok)

	// default state
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// send values
	subject.Next(21)
	subject.Next(42)
	is.Equal(KindNext, subject.status)
	is.Empty(subject.err)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// trigger error
	subject.Error(assert.AnError)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)

	// no change
	subject.Next(84)
	is.Equal(KindError, subject.status)
	is.Equal(assert.AnError, subject.err.B)
	is.Equal(0, syncMapLength(&subject.observers))
	is.Equal(uint32(0), subject.observerIndex)
}

func TestPublishSubject_internalSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

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

func TestPublishSubject_hasObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

	is.True(ok)

	// default state
	is.False(subject.HasObserver())
	subscription := subject.Subscribe(OnNext(func(value int) {}))
	is.True(subject.HasObserver())
	subscription.Unsubscribe()
	is.False(subject.HasObserver())
}

func TestPublishSubject_hasThrown(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewPublishSubject[int]().(*publishSubjectImpl[int])
	subject2, ok2 := NewPublishSubject[int]().(*publishSubjectImpl[int])
	subject3, ok3 := NewPublishSubject[int]().(*publishSubjectImpl[int])

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

func TestPublishSubject_isComplete(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject1, ok1 := NewPublishSubject[int]().(*publishSubjectImpl[int])
	subject2, ok2 := NewPublishSubject[int]().(*publishSubjectImpl[int])
	subject3, ok3 := NewPublishSubject[int]().(*publishSubjectImpl[int])

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

func TestPublishSubject_singleSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])
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

func TestPublishSubject_multipleSubscription(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])
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

func TestPublishSubject_subscriptionCanceledTwice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])
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

func TestPublishSubject_next(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 600*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

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

	time.Sleep(100 * time.Millisecond)
	subject.Next(21)
	time.Sleep(100 * time.Millisecond)
	is.Equal(int64(21), atomic.LoadInt64(&counter1))
	is.Equal(int64(21), atomic.LoadInt64(&counter2))
	is.Equal(int64(21), atomic.LoadInt64(&counter3))

	time.Sleep(100 * time.Millisecond)
	subject.Next(42)
	time.Sleep(100 * time.Millisecond)
	is.Equal(int64(63), atomic.LoadInt64(&counter1))
	is.Equal(int64(63), atomic.LoadInt64(&counter2))
	is.Equal(int64(63), atomic.LoadInt64(&counter3))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
}

func TestPublishSubject_error(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

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
	is.Equal(int64(0), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestPublishSubject_complete(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 200*time.Millisecond)
	is := assert.New(t)

	subject, ok := NewPublishSubject[int]().(*publishSubjectImpl[int])

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
	is.Equal(int64(0), atomic.LoadInt64(&counter4))

	// unsubscribe all
	subscription1.Unsubscribe()
	subscription2.Unsubscribe()
	subscription3.Unsubscribe()
	subscription4.Unsubscribe()
}

func TestPublishSubject_reentrantCallFromObserver(t *testing.T) {
	// @TODO: Known bug. NextWithContext holds s.mu while it runs observer
	// callbacks, so an observer calling back into the subject (here IsClosed)
	// deadlocks. Unskip once the subject stops holding its lock across user code.
	t.Skip("known deadlock: publish subject holds its mutex while calling observers")

	t.Parallel()
	is := assert.New(t)

	subject := NewPublishSubject[int]()
	subject.Subscribe(OnNext(func(int) {
		_ = subject.IsClosed() // takes s.mu again
	}))

	is.True(returnsWithin(500*time.Millisecond, func() { subject.Next(1) }), "Next deadlocked on re-entrant IsClosed")
}

func TestPublishSubject_panickingObserverKeepsSubjectUsable(t *testing.T) {
	// @TODO: Known bug. NextWithContext releases s.mu with a plain Unlock, so a
	// panic raised by an observer leaves the subject locked forever. Unskip once
	// the unlock is deferred.
	t.Skip("known bug: a panicking observer leaves the publish subject mutex locked")

	t.Parallel()
	is := assert.New(t)

	subject := NewPublishSubject[int]()
	subject.Subscribe(panickingObserver{})

	is.Panics(func() { subject.Next(1) })
	is.True(returnsWithin(500*time.Millisecond, func() { _ = subject.IsClosed() }), "mutex still locked after observer panic")
}
