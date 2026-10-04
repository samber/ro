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
	"sync"
	"testing"

	"github.com/samber/ro"
)

// progress is what a stop function needs to know about a stream: *recorder implements it.
type progress interface {
	valueCount() int
	terminalCount() int
}

// backgroundSubscription subscribes from its own goroutine, so that a Subscribe that never returns
// fails the test through expectReturn instead of hanging it.
type backgroundSubscription struct {
	mu           sync.Mutex
	subscription ro.Subscription
	returned     chan struct{}
}

func subscribeInBackground[T any](ctx context.Context, observable ro.Observable[T], observer *recorder[T]) *backgroundSubscription {
	background := &backgroundSubscription{returned: make(chan struct{})}

	go func() {
		subscription := observable.SubscribeWithContext(ctx, observer)

		background.mu.Lock()
		background.subscription = subscription
		background.mu.Unlock()

		close(background.returned)
	}()

	return background
}

func (b *backgroundSubscription) hasReturned() bool { return isClosed(b.returned) }

// expectReturn fails the test when Subscribe does not return in time.
func (b *backgroundSubscription) expectReturn(t *testing.T) {
	t.Helper()

	waitUntil(t, "Subscribe to return (it hangs)", b.hasReturned)
}

// unsubscribe unsubscribes once Subscribe returned. It does nothing before.
func (b *backgroundSubscription) unsubscribe() {
	b.mu.Lock()
	subscription := b.subscription
	b.mu.Unlock()

	if subscription != nil {
		subscription.Unsubscribe()
	}
}

// unsubscribeAfterItems waits until Subscribe returned and either the stream received items values or
// terminated, then unsubscribes from outside, like a consumer losing interest.
func (b *backgroundSubscription) unsubscribeAfterItems(t *testing.T, stream progress, items int) {
	t.Helper()

	waitUntil(t, "Subscribe to return and the stream to reach the unsubscribe point", func() bool {
		return b.hasReturned() && (stream.valueCount() >= items || stream.terminalCount() > 0)
	})

	b.unsubscribe()
}

// cancelAfterItems waits until the stream received items values or terminated, then cancels the
// subscriber context.
func cancelAfterItems(t *testing.T, cancel context.CancelFunc, stream progress, items int) {
	t.Helper()

	waitUntil(t, "the stream to reach the cancellation point", func() bool {
		return stream.valueCount() >= items || stream.terminalCount() > 0
	})

	cancel()
}
