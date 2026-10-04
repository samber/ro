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
	"sync/atomic"
	"testing"
)

// countingPredicate wraps a user predicate to count how often an operator calls it, and how often
// after the call that decided the stream.
type countingPredicate struct {
	// decisiveResult is the result that ends the stream: true for First or Find, false for All or TakeWhile.
	decisiveResult bool
	decide         func(item int) bool

	calls              int32
	decided            int32
	callsAfterDecision int32
}

func newCountingPredicate(decisiveResult bool, decide func(item int) bool) *countingPredicate {
	return &countingPredicate{decisiveResult: decisiveResult, decide: decide}
}

// test is the predicate to give to an operator.
func (p *countingPredicate) test(item int) bool {
	atomic.AddInt32(&p.calls, 1)

	if atomic.LoadInt32(&p.decided) != 0 {
		atomic.AddInt32(&p.callsAfterDecision, 1)
	}

	result := p.decide(item)
	if result == p.decisiveResult {
		atomic.StoreInt32(&p.decided, 1)
	}

	return result
}

// testIndexed is the predicate to give to the operators with an index parameter.
func (p *countingPredicate) testIndexed(item int, _ int64) bool { return p.test(item) }

func (p *countingPredicate) expectNoCallAfterDecision(t *testing.T) {
	t.Helper()

	if extra := atomic.LoadInt32(&p.callsAfterDecision); extra != 0 {
		t.Fatalf("predicate called %d times after the call that decided the stream (%d calls in total)",
			extra, atomic.LoadInt32(&p.calls))
	}
}
