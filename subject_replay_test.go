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
