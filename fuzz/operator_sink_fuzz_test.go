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
	"errors"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
	"github.com/samber/ro/internal/xfuzz"
	"github.com/stretchr/testify/assert"
)

func FuzzTimeToChannel(f *testing.F) {
	fuzzTimeSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask, k uint8) {
		sc := fuzzTimeDecode(seed, mask, k, true)

		var counter activeCounter

		fuzzTimeIter(t, "ToChannel", func() error {
			size := fuzzTimePick(seed, 12, 0, 4)
			src := trackSubscriptions(&counter, fuzzSource(seed, sc.items, sc.async))
			obs := ro.ToChannel[int](size)(src)

			var (
				channels      int32
				completes     int32
				completeFirst int32
				consumed      int32
				sawTerminal   int32
			)

			consumerDone := make(chan struct{})

			slow := fuzzTimeMicros(seed, 13, 0) // delays Next(ch), giving Complete the chance to win.

			obs.Subscribe(ro.NewObserver(
				func(ch <-chan ro.Notification[int]) {
					atomic.AddInt32(&channels, 1)
					time.Sleep(slow)

					go func() {
						defer close(consumerDone)

						for n := range ch {
							if n.Kind == ro.KindNext {
								atomic.AddInt32(&consumed, 1)
							} else {
								atomic.AddInt32(&sawTerminal, 1)
							}
						}
					}()
				},
				func(error) {},
				func() {
					if atomic.LoadInt32(&channels) == 0 {
						atomic.AddInt32(&completeFirst, 1)
					}

					atomic.AddInt32(&completes, 1)
				},
			))

			deadline := time.After(fuzzDeadline / 2)

			select {
			case <-consumerDone:
			case <-deadline:
				return fmt.Errorf("channel never closed: consumed %d/%d, channels=%d completes=%d", atomic.LoadInt32(&consumed), sc.items, atomic.LoadInt32(&channels), atomic.LoadInt32(&completes))
			}

			fuzzTimeWaitPlain(func() bool { return atomic.LoadInt32(&completes) > 0 })

			switch {
			case atomic.LoadInt32(&completeFirst) > 0:
				return errors.New("Complete delivered before the channel was emitted")
			case atomic.LoadInt32(&channels) != 1:
				return fmt.Errorf("channel emitted %d times", atomic.LoadInt32(&channels))
			case atomic.LoadInt32(&consumed) != int32(sc.items): //nolint:gosec // bounded.
				return fmt.Errorf("consumer got %d of %d items", atomic.LoadInt32(&consumed), sc.items)
			case atomic.LoadInt32(&sawTerminal) != 1:
				return fmt.Errorf("consumer saw %d terminal notifications", atomic.LoadInt32(&sawTerminal))
			}

			return nil
		})

		fuzzTimeWaitUpstreamClosed(t, &counter)
	})
}

// fuzzTimeWaitPlain polls cond for at most one second and reports whether it became true.
func fuzzTimeWaitPlain(cond func() bool) bool {
	deadline := time.Now().Add(time.Second)
	for !cond() {
		if time.Now().After(deadline) {
			return false
		}

		time.Sleep(time.Millisecond)
	}

	return true
}

// A source that keeps emitting after the downstream unsubscribed must not
// make ToChannel send on a closed channel.
func FuzzOperatorSinkToChannelUnsubscribeWhileSending(f *testing.F) {
	xfuzz.AddSeeds(f, func(i int) []any { return []any{int64(i)} })

	f.Fuzz(func(t *testing.T, seed int64) {
		testWithTimeout(t, 2*time.Second)
		is := assert.New(t)

		senderDone := make(chan struct{})
		source := ro.NewUnsafeObservable(func(destination ro.Observer[int]) ro.Teardown {
			go func() {
				defer close(senderDone)
				for i := 0; i < 100; i++ {
					destination.Next(i)
				}
				destination.Complete()
			}()
			return nil
		})

		received := make(chan (<-chan ro.Notification[int]), 1)
		sub := ro.ToChannel[int](0)(source).Subscribe(ro.OnNext(func(ch <-chan ro.Notification[int]) {
			received <- ch
		}))

		ch := <-received
		// Nobody reads: the source goroutine blocks on a full channel.
		time.Sleep(5 * time.Millisecond)
		fuzzJitter(seed, 0)
		sub.Unsubscribe()

		select {
		case <-senderDone:
		case <-time.After(time.Second):
			is.Fail("source goroutine still blocked after unsubscribe")
		}

		// The channel must end up closed: draining it only terminates once it is.
		drained := 0
		for range ch {
			drained++
		}
		is.GreaterOrEqual(drained, 0)
	})
}
