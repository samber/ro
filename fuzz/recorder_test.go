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

package fuzz

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/samber/ro"
)

// recorder is a destination that keeps everything it receives, including the notifications sent
// after a terminal one, and detects overlapping callbacks. Its IsClosed is always false, so the
// operator under test decides on its own when to stop.
//
// It never stores a testing.T: the expect methods take the test of the calling goroutine.
type recorder[T any] struct {
	guard overlapGuard

	mu                   sync.Mutex
	values               []T
	errors               []error
	completions          int
	notifsAfterTerminals int
}

func newRecorder[T any]() *recorder[T] { return &recorder[T]{} }

// noteAfterTerminal must run with mu held, before the notification is stored.
func (r *recorder[T]) noteAfterTerminal() {
	if len(r.errors)+r.completions > 0 {
		r.notifsAfterTerminals++
	}
}

func (r *recorder[T]) Next(value T) { r.NextWithContext(context.Background(), value) }

func (r *recorder[T]) NextWithContext(_ context.Context, value T) {
	r.guard.enter()
	defer r.guard.leave()

	r.mu.Lock()
	defer r.mu.Unlock()

	r.noteAfterTerminal()
	r.values = append(r.values, value)
}

func (r *recorder[T]) Error(err error) { r.ErrorWithContext(context.Background(), err) }

func (r *recorder[T]) ErrorWithContext(_ context.Context, err error) {
	r.guard.enter()
	defer r.guard.leave()

	r.mu.Lock()
	defer r.mu.Unlock()

	r.noteAfterTerminal()
	r.errors = append(r.errors, err)
}

func (r *recorder[T]) Complete() { r.CompleteWithContext(context.Background()) }

func (r *recorder[T]) CompleteWithContext(_ context.Context) {
	r.guard.enter()
	defer r.guard.leave()

	r.mu.Lock()
	defer r.mu.Unlock()

	r.noteAfterTerminal()
	r.completions++
}

func (r *recorder[T]) IsClosed() bool { return false }

func (r *recorder[T]) HasThrown() bool { return false }

func (r *recorder[T]) IsCompleted() bool { return false }

// received returns a copy of the values received so far.
func (r *recorder[T]) received() []T {
	r.mu.Lock()
	defer r.mu.Unlock()

	return append([]T(nil), r.values...)
}

func (r *recorder[T]) valueCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.values)
}

func (r *recorder[T]) terminalCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()

	return len(r.errors) + r.completions
}

// expectValues checks the exact values received, in order. A nil and an empty slice are equal.
func (r *recorder[T]) expectValues(t *testing.T, want []T) {
	t.Helper()

	got := r.received()
	if len(got) == 0 && len(want) == 0 {
		return
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

// expectAtMostValues checks that no more than limit values were received.
func (r *recorder[T]) expectAtMostValues(t *testing.T, limit int) {
	t.Helper()

	if got := r.received(); len(got) > limit {
		t.Fatalf("values = %v, want at most %d", got, limit)
	}
}

// expectCompletedOnce checks that the stream completed exactly once, without error.
func (r *recorder[T]) expectCompletedOnce(t *testing.T) {
	t.Helper()

	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.errors) != 0 || r.completions != 1 {
		t.Fatalf("terminals: errors=%v completions=%d, want no error and exactly one completion", r.errors, r.completions)
	}
}

// expectFailedOnce checks that the stream failed exactly once, without completing.
func (r *recorder[T]) expectFailedOnce(t *testing.T) {
	t.Helper()

	r.mu.Lock()
	defer r.mu.Unlock()

	if len(r.errors) != 1 || r.completions != 0 {
		t.Fatalf("terminals: errors=%v completions=%d, want exactly one error and no completion", r.errors, r.completions)
	}
}

// expectContract checks the observer contract: no overlapping callbacks, at most one terminal
// notification, and no notification after it.
func (r *recorder[T]) expectContract(t *testing.T) {
	t.Helper()

	r.guard.expectNone(t)

	r.mu.Lock()
	defer r.mu.Unlock()

	if terminals := len(r.errors) + r.completions; terminals > 1 {
		t.Fatalf("%d terminal notifications (errors=%v completions=%d)", terminals, r.errors, r.completions)
	}

	if r.notifsAfterTerminals != 0 {
		t.Fatalf("%d notifications received after a terminal notification", r.notifsAfterTerminals)
	}
}

// collect subscribes to observable, waits for the producer goroutines of the given sources, gives stray
// notifications settleDelay to arrive, unsubscribes, and returns what the destination saw.
func collect[T any](t *testing.T, observable ro.Observable[T], sources ...*source) *recorder[T] {
	t.Helper()

	got := newRecorder[T]()

	runWithinDeadline(t, func() {
		subscription := observable.SubscribeWithContext(context.Background(), got)

		for _, s := range sources {
			s.waitForProducers()
		}

		time.Sleep(settleDelay)

		subscription.Unsubscribe()
	})

	return got
}

// droppedNotifications counts the notifications that a Subscriber absorbed because it was already
// closed. They are how a double emission by an operator shows up when the destination is a Subscriber.
type droppedNotifications struct {
	mu    sync.Mutex
	count int
}

// captureDroppedNotifications hooks ro.OnDroppedNotification until the end of the test. The hook is a
// package-level variable, so a target using it must not run in parallel.
func captureDroppedNotifications(t *testing.T) *droppedNotifications {
	t.Helper()

	dropped := &droppedNotifications{}
	previous := ro.OnDroppedNotification

	ro.OnDroppedNotification = func(context.Context, fmt.Stringer) {
		dropped.mu.Lock()
		dropped.count++
		dropped.mu.Unlock()
	}

	t.Cleanup(func() { ro.OnDroppedNotification = previous })

	return dropped
}

func (d *droppedNotifications) expectNone(t *testing.T) {
	t.Helper()

	d.mu.Lock()
	defer d.mu.Unlock()

	if d.count != 0 {
		t.Fatalf("%d notifications dropped by a closed subscriber", d.count)
	}
}
