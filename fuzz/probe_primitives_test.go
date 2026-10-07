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
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/samber/ro"
)

// maxWorkers bounds the number of concurrent goroutines or subscribers a fuzz input can request.
const maxWorkers = 8

// errInjectedFailure is the error sources and subjects end with when a target asks for a failing stream.
var errInjectedFailure = errors.New("fuzz: injected failure")

// subscriberProbe is a subscriber that records what it receives and checks the observer contract:
// no overlapping callbacks, at most one terminal notification, nothing after it. Two hooks let a
// target act from inside a callback, which is where re-entrancy bugs hide.
type subscriberProbe struct {
	guard overlapGuard

	valueTotal       int32
	errorTotal       int32
	completionTotal  int32
	afterTerminal    int32
	afterUnsubscribe int32
	unsubscribed     int32

	hookFired int32
	// onFirstNotification runs once, inside the first callback of any kind.
	onFirstNotification func()
	// onValue runs inside every Next callback, after the value was counted.
	onValue func(value int)

	mu           sync.Mutex
	valuesSeen   []int
	subscription ro.Subscription
}

// subscribe subscribes the probe to source and remembers the subscription, so that unsubscribe can
// end it from any goroutine. A callback running before Subscribe returns sees no subscription yet.
func (p *subscriberProbe) subscribe(source ro.Observable[int]) ro.Subscription {
	subscription := source.Subscribe(p.observer())

	p.mu.Lock()
	p.subscription = subscription
	p.mu.Unlock()

	return subscription
}

// unsubscribe ends the subscription made by subscribe, then marks the probe so that any later
// notification counts as delivered after the unsubscription. It does nothing before subscribe returned.
func (p *subscriberProbe) unsubscribe() {
	p.mu.Lock()
	subscription := p.subscription
	p.mu.Unlock()

	if subscription == nil {
		return
	}

	subscription.Unsubscribe()
	p.markUnsubscribed()
}

// markUnsubscribed tells the probe that the target unsubscribed its subscriber.
func (p *subscriberProbe) markUnsubscribed() { atomic.StoreInt32(&p.unsubscribed, 1) }

func (p *subscriberProbe) valueCount() int { return int(atomic.LoadInt32(&p.valueTotal)) }

func (p *subscriberProbe) terminalCount() int {
	return int(atomic.LoadInt32(&p.errorTotal) + atomic.LoadInt32(&p.completionTotal))
}

// received returns a copy of the values received so far.
func (p *subscriberProbe) received() []int {
	p.mu.Lock()
	defer p.mu.Unlock()

	return append([]int(nil), p.valuesSeen...)
}

func (p *subscriberProbe) enter() {
	p.guard.enter()

	if p.terminalCount() > 0 {
		atomic.AddInt32(&p.afterTerminal, 1)
	}

	if atomic.LoadInt32(&p.unsubscribed) == 1 {
		atomic.AddInt32(&p.afterUnsubscribe, 1)
	}

	if p.onFirstNotification != nil && atomic.CompareAndSwapInt32(&p.hookFired, 0, 1) {
		p.onFirstNotification()
	}
}

func (p *subscriberProbe) observer() ro.Observer[int] {
	return ro.NewObserver(
		func(value int) {
			p.enter()
			defer p.guard.leave()

			atomic.AddInt32(&p.valueTotal, 1)

			p.mu.Lock()
			p.valuesSeen = append(p.valuesSeen, value)
			p.mu.Unlock()

			if p.onValue != nil {
				p.onValue(value)
			}
		},
		func(error) {
			p.enter()
			defer p.guard.leave()

			atomic.AddInt32(&p.errorTotal, 1)
		},
		func() {
			p.enter()
			defer p.guard.leave()

			atomic.AddInt32(&p.completionTotal, 1)
		},
	)
}

// expectContract checks the invariants every subscriber must see, whatever the interleaving. who names
// the subscriber in the failure message.
func (p *subscriberProbe) expectContract(t *testing.T, who string) {
	t.Helper()

	if got := p.terminalCount(); got > 1 {
		t.Fatalf("%s: %d terminal notifications, want at most 1", who, got)
	}

	if got := atomic.LoadInt32(&p.afterTerminal); got != 0 {
		t.Fatalf("%s: %d notifications delivered after a terminal", who, got)
	}

	if got := p.guard.overlapCount(); got != 0 {
		t.Fatalf("%s: %d overlapping notifications", who, got)
	}
}

// expectSilentAfterUnsubscribe checks that nothing reached the probe once it was marked unsubscribed.
func (p *subscriberProbe) expectSilentAfterUnsubscribe(t *testing.T, who string) {
	t.Helper()

	if got := atomic.LoadInt32(&p.afterUnsubscribe); got != 0 {
		t.Fatalf("%s: %d notifications after it unsubscribed", who, got)
	}
}

// waitGroupFinished reports whether group finished within timeout, without failing the test.
func waitGroupFinished(group *sync.WaitGroup, timeout time.Duration) bool {
	finished := make(chan struct{})

	go func() {
		group.Wait()
		close(finished)
	}()

	select {
	case <-finished:
		return true
	case <-time.After(timeout):
		return false
	}
}

// expectWaitGroupDone fails the test when group does not finish within waitDeadline, so that a
// deadlock fails the target instead of hanging the run.
func expectWaitGroupDone(t *testing.T, what string, group *sync.WaitGroup) {
	t.Helper()

	if !waitGroupFinished(group, waitDeadline) {
		t.Fatalf("deadlock: %s did not finish within %s", what, waitDeadline)
	}
}

// expectReturns runs call in its own goroutine and fails the test when it does not return within waitDeadline.
func expectReturns(t *testing.T, what string, call func()) {
	t.Helper()

	var group sync.WaitGroup

	group.Add(1)

	go func() {
		defer group.Done()

		call()
	}()

	expectWaitGroupDone(t, what, &group)
}
