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
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// channelHandoffUnit is the granularity of the pause before a ToChannel consumer starts reading. With
// a uint8 input it keeps the longest pause near 2 ms, long enough for the completion of a short source
// to arrive before the consumer reads anything.
const channelHandoffUnit = 8 * time.Microsecond

// channelReader is a downstream of ToChannel: it receives the channel, reads every notification
// from a goroutine of its own, and counts what it saw.
type channelReader struct {
	handoffDelay time.Duration

	channelsReceived int32
	completions      int32
	completedFirst   int32
	valuesRead       int32
	terminalsRead    int32
	channelDrained   chan struct{}
}

func newChannelReader(handoffDelay time.Duration) *channelReader {
	return &channelReader{handoffDelay: handoffDelay, channelDrained: make(chan struct{})}
}

func (r *channelReader) observer() ro.Observer[<-chan ro.Notification[int]] {
	return ro.NewObserver(
		r.receiveChannel,
		func(error) {},
		func() {
			// The channel is emitted before the source is subscribed: a completion that
			// arrives first means the operator reordered its notifications.
			if atomic.LoadInt32(&r.channelsReceived) == 0 {
				atomic.AddInt32(&r.completedFirst, 1)
			}

			atomic.AddInt32(&r.completions, 1)
		},
	)
}

func (r *channelReader) receiveChannel(notifications <-chan ro.Notification[int]) {
	atomic.AddInt32(&r.channelsReceived, 1)

	// A slow handoff gives the completion the chance to win the race against the channel.
	time.Sleep(r.handoffDelay)

	go func() {
		defer close(r.channelDrained)

		for notification := range notifications {
			if notification.Kind == ro.KindNext {
				atomic.AddInt32(&r.valuesRead, 1)
			} else {
				atomic.AddInt32(&r.terminalsRead, 1)
			}
		}
	}()
}

// expectWholeStreamRead waits for the channel to be closed, then checks that the consumer got the
// channel once, every item, and one terminal notification, and that the completion came after the channel.
func (r *channelReader) expectWholeStreamRead(t *testing.T, items int) {
	t.Helper()

	select {
	case <-r.channelDrained:
	case <-time.After(waitDeadline / 2):
		t.Fatalf("channel never closed: read %d/%d items, channels=%d completions=%d",
			atomic.LoadInt32(&r.valuesRead), items, atomic.LoadInt32(&r.channelsReceived), atomic.LoadInt32(&r.completions))
	}

	waitUntil(t, "the completion of the downstream", func() bool { return atomic.LoadInt32(&r.completions) > 0 })

	if atomic.LoadInt32(&r.completedFirst) > 0 {
		t.Fatal("Complete delivered before the channel was emitted")
	}

	if got := atomic.LoadInt32(&r.channelsReceived); got != 1 {
		t.Fatalf("channel emitted %d times, want once", got)
	}

	if got := int(atomic.LoadInt32(&r.valuesRead)); got != items {
		t.Fatalf("consumer got %d of %d items", got, items)
	}

	if got := atomic.LoadInt32(&r.terminalsRead); got != 1 {
		t.Fatalf("consumer saw %d terminal notifications, want 1", got)
	}
}

// FuzzToChannel turns a source of `items` integers into a channel of notifications, and reads the channel
// from a goroutine of its own. The channel has a fuzzed buffer size, the source is synchronous or
// asynchronous, and the consumer starts reading after a fuzzed pause.
//
// Invariant: the channel is emitted once, before any completion; the consumer reads every item and
// exactly one terminal notification, then the channel closes; the source subscription is released.
//
// Seeds: items, bufferSize and handoffDelay spread over their range; asyncSource alternates.
func FuzzToChannel(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// items, bufferSize, asyncSource, handoffDelay (in units of channelHandoffUnit)
		return []any{seedByte(i, 0), seedByte(i, 1), i%2 == 0, seedByte(i, 2)}
	})

	f.Fuzz(func(t *testing.T, items, bufferSize uint8, asyncSource bool, handoffDelay uint8) {
		count := bounded(items, 1, maxItems)
		buffer := bounded(bufferSize, 0, 4)

		numbers := newSource(count, asyncSource)
		upstream := &subscriptionCounter{}
		reader := newChannelReader(time.Duration(handoffDelay) * channelHandoffUnit)

		runWithinDeadline(t, func() {
			ro.ToChannel[int](buffer)(countSubscriptions(upstream, numbers.observable())).Subscribe(reader.observer())
		})

		reader.expectWholeStreamRead(t, count)
		upstream.expectAllReleased(t)
	})
}

// blockedSenderPause lets the source goroutine fill the channel buffer and block, before the unsubscription.
const blockedSenderPause = 5 * time.Millisecond

// sendingBurst is the number of items a blocked sender wants to emit: more than any buffer size.
const sendingBurst = 100

// sourceIgnoringUnsubscribe emits sendingBurst items from a goroutine that never looks at
// unsubscription, and closes senderDone when it returns.
func sourceIgnoringUnsubscribe(senderDone chan struct{}) ro.Observable[int] {
	return ro.NewUnsafeObservableWithContext(func(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
		go func() {
			defer close(senderDone)

			for item := 0; item < sendingBurst; item++ {
				destination.NextWithContext(ctx, item)
			}

			destination.CompleteWithContext(ctx)
		}()

		return nil
	})
}

// FuzzToChannelUnsubscribeWhileSending unsubscribes from ToChannel while its source goroutine is blocked
// sending into a channel that nobody reads. The source keeps emitting after the unsubscription.
//
// Invariant: the source goroutine is released by the Unsubscribe, without a panic from a send on a closed
// channel, and the channel ends up closed.
//
// Seeds: bufferSize and yieldPattern spread over their range.
func FuzzToChannelUnsubscribeWhileSending(f *testing.F) {
	fuzzSeeds(f, func(i int) []any {
		// bufferSize, yieldPattern
		return []any{seedByte(i, 0), seedByte(i, 1)}
	})

	f.Fuzz(func(t *testing.T, bufferSize, yieldPattern uint8) {
		buffer := bounded(bufferSize, 0, 4)

		senderDone := make(chan struct{})
		emitted := make(chan (<-chan ro.Notification[int]), 1)

		subscription := ro.ToChannel[int](buffer)(sourceIgnoringUnsubscribe(senderDone)).Subscribe(
			ro.OnNext(func(notifications <-chan ro.Notification[int]) { emitted <- notifications }),
		)

		notifications := <-emitted

		// Nobody reads: once the buffer is full, the source goroutine blocks in Next.
		time.Sleep(blockedSenderPause)
		yieldAt(int64(yieldPattern), 0)
		subscription.Unsubscribe()

		select {
		case <-senderDone:
		case <-time.After(time.Second):
			t.Fatal("source goroutine still blocked after unsubscribe")
		}

		// Draining only terminates once the channel is closed.
		runWithinDeadline(t, func() {
			for range notifications { //nolint:revive // draining is the whole point.
			}
		})
	})
}
