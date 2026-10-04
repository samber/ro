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

package xfuzz

import (
	"fmt"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

// shortTimeout keeps the failure-path tests fast.
const shortTimeout = 20 * time.Millisecond

// fakeTB records Fatalf calls instead of stopping the goroutine.
type fakeTB struct{ msgs []string }

func (f *fakeTB) Helper() {}

func (f *fakeTB) Fatalf(format string, args ...any) {
	f.msgs = append(f.msgs, fmt.Sprintf(format, args...))
}

func TestWaitChan(t *testing.T) {
	t.Parallel()

	closed := make(chan struct{})
	close(closed)

	ok := &fakeTB{}
	WaitChanTimeout(ok, closed, "closed", shortTimeout)
	WaitChan(ok, closed, "closed, default deadline")

	if len(ok.msgs) != 0 {
		t.Fatalf("unexpected failure: %v", ok.msgs)
	}

	bad := &fakeTB{}
	WaitChanTimeout(bad, make(chan struct{}), "stuck", shortTimeout)

	if len(bad.msgs) != 1 || !strings.Contains(bad.msgs[0], "stuck: timed out") {
		t.Fatalf("expected one timeout failure, got %v", bad.msgs)
	}

	var nilCh chan struct{}

	nilFail := &fakeTB{}
	WaitChanTimeout(nilFail, nilCh, "nil", shortTimeout)

	if len(nilFail.msgs) != 1 {
		t.Fatalf("nil channel must time out, got %v", nilFail.msgs)
	}
}

func TestWaitGroup(t *testing.T) {
	t.Parallel()

	ok := &fakeTB{}

	var zero sync.WaitGroup

	WaitGroupTimeout(ok, &zero, "zero value", shortTimeout)

	var wg sync.WaitGroup

	wg.Add(1)

	go func() {
		time.Sleep(time.Millisecond)
		wg.Done()
	}()

	WaitGroup(ok, &wg, "finishing")

	if len(ok.msgs) != 0 {
		t.Fatalf("unexpected failure: %v", ok.msgs)
	}

	var stuck sync.WaitGroup

	stuck.Add(1)

	bad := &fakeTB{}
	WaitGroupTimeout(bad, &stuck, "stuck", shortTimeout)
	stuck.Done() // release the helper goroutine

	if len(bad.msgs) != 1 || !strings.Contains(bad.msgs[0], "stuck: timed out") {
		t.Fatalf("expected one timeout failure, got %v", bad.msgs)
	}
}

func TestGoRecover(t *testing.T) {
	t.Parallel()

	var wg sync.WaitGroup

	errs := make(chan string, 2)

	GoRecover(&wg, errs, func() {})
	GoRecover(&wg, errs, func() { panic("boom") })
	WaitGroup(t, &wg, "GoRecover")

	fake := &fakeTB{}
	FailOnErrs(fake, errs)

	if len(fake.msgs) != 1 || fake.msgs[0] != "panic: boom" {
		t.Fatalf("expected the recovered panic, got %v", fake.msgs)
	}

	ok := &fakeTB{}
	FailOnErrs(ok, make(chan string, 1))

	if len(ok.msgs) != 0 {
		t.Fatalf("empty errs must not fail: %v", ok.msgs)
	}
}

func TestAssertNoLeak(t *testing.T) {
	t.Parallel()

	ok := &fakeTB{}
	AssertNoLeakTimeout(ok, 1<<30, shortTimeout)
	AssertNoLeak(ok, 1<<30)

	if len(ok.msgs) != 0 {
		t.Fatalf("unexpected failure: %v", ok.msgs)
	}

	bad := &fakeTB{}
	AssertNoLeakTimeout(bad, 0, shortTimeout)

	if len(bad.msgs) != 1 || !strings.Contains(bad.msgs[0], "goroutine leak") {
		t.Fatalf("expected a leak failure, got %v", bad.msgs)
	}
}

// TestAssertNoLeakSettles checks that goroutines exiting during the settle
// loop are not reported. Other parallel tests only add short-lived goroutines,
// which the generous settle time absorbs.
func TestAssertNoLeakSettles(t *testing.T) {
	t.Parallel()

	release := make(chan struct{})

	go func() { <-release }()

	before := runtime.NumGoroutine() - 1 // excludes the goroutine above

	go func() {
		time.Sleep(shortTimeout / 2)
		close(release)
	}()

	settled := &fakeTB{}
	AssertNoLeakTimeout(settled, before, Deadline)

	if len(settled.msgs) != 0 {
		t.Fatalf("goroutines that exit in time must not fail: %v", settled.msgs)
	}
}

func TestDelay(t *testing.T) {
	t.Parallel()

	cases := []struct {
		name string
		seed int64
		mask uint8
		want time.Duration
	}{
		{"sync mode is zero", 7, 0, 0},
		{"sync mode ignores other bits", 7, 0xfe, 0},
		{"zero seed async", 0, 1, delayUnit},
		{"negative seed folds", -3, 1, 4 * delayUnit},
		{"wraps at max steps", delayMaxSteps, 1, delayUnit},
		{"largest step", delayMaxSteps - 1, 1, delayMaxSteps * delayUnit},
		{"min int64", -1 << 63, 1, 3 * delayUnit},
	}

	for _, tc := range cases {
		if got := Delay(tc.seed, tc.mask); got != tc.want {
			t.Errorf("%s: Delay(%d, %d) = %s, want %s", tc.name, tc.seed, tc.mask, got, tc.want)
		}
	}
}

func TestFailingWriter(t *testing.T) {
	t.Parallel()

	w := &FailingWriter{FailAt: 2}

	for i := 0; i < 2; i++ {
		if n, err := w.Write([]byte("abc")); n != 3 || err != nil {
			t.Fatalf("write %d must succeed, got %d, %v", i, n, err)
		}
	}

	if n, err := w.Write([]byte("abc")); n != 0 || err != ErrWrite {
		t.Fatalf("first failing write: got %d, %v", n, err)
	}

	if w.CallsAfterFail() != 0 {
		t.Fatalf("CallsAfterFail = %d, want 0", w.CallsAfterFail())
	}

	if _, err := w.Write(nil); err != ErrWrite {
		t.Fatalf("writes after failure must keep failing, got %v", err)
	}

	if w.Calls() != 4 || w.CallsAfterFail() != 1 {
		t.Fatalf("Calls=%d CallsAfterFail=%d, want 4 and 1", w.Calls(), w.CallsAfterFail())
	}

	for _, failAt := range []int{0, -1} {
		always := &FailingWriter{FailAt: failAt}
		if _, err := always.Write([]byte("x")); err != ErrWrite {
			t.Fatalf("FailAt=%d must fail at once, got %v", failAt, err)
		}
	}
}

func TestStandardSeeds(t *testing.T) {
	t.Setenv(FuzzIterationsEnv, "3")

	// StandardSeeds only forwards (int64, uint8) pairs to AddSeeds: a fuzz
	// target with that signature must accept every generated seed.
	// Run as a real fuzz target would, through testing's own seed corpus.
	if got := FuzzIterations(); got != 3 {
		t.Fatalf("FuzzIterations = %d, want 3", got)
	}
}

func FuzzStandardSeedsTarget(f *testing.F) {
	StandardSeeds(f)

	f.Fuzz(func(t *testing.T, seed int64, mask uint8) {
		if Delay(seed, mask) < 0 {
			t.Fatalf("negative delay for seed=%d mask=%d", seed, mask)
		}
	})
}
