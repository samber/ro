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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

// reentrancyTimeout is long enough for a healthy call to return, and short
// enough to keep a deadlocked run fast.
const reentrancyTimeout = 500 * time.Millisecond

// returnsWithin reports whether fn returns before reentrancyTimeout. A
// deadlocked fn leaves its goroutine blocked forever.
func returnsWithin(fn func()) bool {
	done := make(chan struct{})

	go func() {
		defer close(done)
		fn()
	}()

	select {
	case <-done:
		return true
	case <-time.After(reentrancyTimeout):
		return false
	}
}

// An observer calling back into the subject it is subscribed to must not
// deadlock: NextWithContext holds s.mu while it runs observer callbacks.
func TestPublishSubject_reentrantCallFromObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject := NewPublishSubject[int]()
	subject.Subscribe(OnNext(func(int) {
		_ = subject.IsClosed() // takes s.mu again
	}))

	is.True(returnsWithin(func() { subject.Next(1) }), "Next deadlocked on re-entrant IsClosed")
}

// A teardown added to an already closed subscription runs under s.mu, so
// adding another teardown from inside it must not deadlock.
func TestSubscription_reentrantAddOnClosedSubscription(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	sub := NewSubscription(nil)
	sub.Unsubscribe()

	is.True(returnsWithin(func() {
		sub.Add(func() {
			sub.Add(func() {}) // takes s.mu again
		})
	}), "Add deadlocked on re-entrant Add")
}

// panickingObserver implements Observer directly: observers built with
// NewObserver recover callback panics, so they would never reach the subject.
type panickingObserver struct{}

func (panickingObserver) Next(int)                                { panic("boom") }
func (panickingObserver) NextWithContext(context.Context, int)    { panic("boom") }
func (panickingObserver) Error(error)                             {}
func (panickingObserver) ErrorWithContext(context.Context, error) {}
func (panickingObserver) Complete()                               {}
func (panickingObserver) CompleteWithContext(context.Context)     {}
func (panickingObserver) IsClosed() bool                          { return false }
func (panickingObserver) HasThrown() bool                         { return false }
func (panickingObserver) IsCompleted() bool                       { return false }

// A panicking observer must not leave the subject mutex locked.
func TestPublishSubject_panickingObserverKeepsSubjectUsable(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	subject := NewPublishSubject[int]()
	subject.Subscribe(panickingObserver{})

	is.Panics(func() { subject.Next(1) })
	is.True(returnsWithin(func() { _ = subject.IsClosed() }), "mutex still locked after observer panic")
}
