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
	"fmt"
	"runtime/debug"
	"testing"
	"time"
)

const (
	// waitDeadline bounds every blocking wait of a fuzz iteration. A hang is a bug, so the bound
	// is generous enough for a loaded CI machine but short enough to fail fast.
	waitDeadline = 5 * time.Second

	// settleDelay lets goroutines that outlive a decision misbehave before post-conditions are read.
	settleDelay = 3 * time.Millisecond
)

// waitUntil polls condition until it is true, and fails the test with "timeout waiting for: <what>"
// once waitDeadline has passed.
func waitUntil(t *testing.T, what string, condition func() bool) {
	t.Helper()

	deadline := time.Now().Add(waitDeadline)
	for !condition() {
		if time.Now().After(deadline) {
			t.Fatalf("timeout waiting for: %s", what)
		}

		time.Sleep(time.Millisecond)
	}
}

// runWithinDeadline runs body in its own goroutine and fails the test when body panics or does not
// return within waitDeadline. body must not call t: a hang then fails the target instead of the CI job.
func runWithinDeadline(t *testing.T, body func()) {
	t.Helper()

	finished := make(chan string, 1)

	go func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				finished <- fmt.Sprintf("panic: %v\n%s", recovered, debug.Stack())
			}
		}()

		body()

		finished <- ""
	}()

	select {
	case failure := <-finished:
		if failure != "" {
			t.Fatal(failure)
		}
	case <-time.After(waitDeadline):
		t.Fatalf("hang: iteration did not finish within %s", waitDeadline)
	}
}
