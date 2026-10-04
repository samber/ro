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
	"errors"
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

const (
	// Deadline bounds every wait of a fuzz target so a deadlock or a hang becomes
	// a failure instead of a stuck CI job. It is far above any healthy scenario
	// (a few milliseconds) yet far below the default `go test` timeout.
	Deadline = 5 * time.Second

	// SettleTime is how long goroutines get to exit before being declared leaked.
	// Teardown is asynchronous, so a strict instant comparison would be flaky.
	SettleTime = 2 * time.Second

	// settlePollInterval is the leak-check polling period: short enough to return
	// quickly on the happy path, long enough not to spin the scheduler.
	settlePollInterval = 5 * time.Millisecond

	// delayModeAsync is the bit of the mask selecting a delayed (async) scenario.
	delayModeAsync = 1 << 0

	// delayUnit scales the derived delay: tiny so an iteration lasts milliseconds.
	delayUnit = 200 * time.Microsecond

	// delayMaxSteps bounds the derived delay at delayMaxSteps*delayUnit.
	delayMaxSteps = 10
)

// ErrWrite is the error returned by FailingWriter once it started failing.
var ErrWrite = errors.New("xfuzz: injected write failure")

// TB is the subset of testing.TB used by the helpers, so they can be unit-tested
// with a fake. *testing.T satisfies it.
type TB interface {
	Helper()
	Fatalf(format string, args ...any)
}

// StandardSeeds registers FuzzIterations() seeds for a target with the common
// signature func(t *testing.T, seed int64, mask uint8).
func StandardSeeds(f *testing.F) {
	f.Helper()

	AddSeeds(f, func(i int) []any { return []any{int64(i), uint8(i)} })
}

// WaitChan fails the target when ch is not closed (or fed) within Deadline.
func WaitChan(t TB, ch <-chan struct{}, what string) {
	t.Helper()

	WaitChanTimeout(t, ch, what, Deadline)
}

// WaitChanTimeout is WaitChan with an explicit timeout.
func WaitChanTimeout(t TB, ch <-chan struct{}, what string, timeout time.Duration) {
	t.Helper()

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	select {
	case <-ch:
	case <-timer.C:
		t.Fatalf("%s: timed out after %s", what, timeout)
	}
}

// WaitGroup fails the target when wg does not finish within Deadline.
func WaitGroup(t TB, wg *sync.WaitGroup, what string) {
	t.Helper()

	WaitGroupTimeout(t, wg, what, Deadline)
}

// WaitGroupTimeout is WaitGroup with an explicit timeout. The helper goroutine
// waiting on wg is left behind on timeout, which is acceptable as the target fails.
func WaitGroupTimeout(t TB, wg *sync.WaitGroup, what string, timeout time.Duration) {
	t.Helper()

	done := make(chan struct{})

	go func() {
		wg.Wait()
		close(done)
	}()

	WaitChanTimeout(t, done, what, timeout)
}

// RecoverInto turns a panic of the calling goroutine into a message on errs,
// since a panic in a goroutine would otherwise kill the whole test binary.
// It must be deferred directly: `defer xfuzz.RecoverInto(errs)`.
func RecoverInto(errs chan<- string) {
	if r := recover(); r != nil {
		errs <- fmt.Sprintf("panic: %v", r)
	}
}

// GoRecover runs fn in a goroutine tracked by wg and reports a recovered panic
// through errs. errs must be buffered enough not to block the sender.
func GoRecover(wg *sync.WaitGroup, errs chan<- string, fn func()) {
	wg.Add(1)

	go func() {
		defer wg.Done()
		defer RecoverInto(errs)

		fn()
	}()
}

// FailOnErrs closes errs, then fails the target with the first message in it.
// No goroutine may send on errs after the call.
func FailOnErrs(t TB, errs chan string) {
	t.Helper()

	close(errs)

	if msg, ok := <-errs; ok {
		t.Fatalf("%s", msg)
	}
}

// AssertNoLeak fails when more goroutines than before (a runtime.NumGoroutine
// snapshot) are still alive after SettleTime.
func AssertNoLeak(t TB, before int) {
	t.Helper()

	AssertNoLeakTimeout(t, before, SettleTime)
}

// AssertNoLeakTimeout is AssertNoLeak with an explicit settle time.
func AssertNoLeakTimeout(t TB, before int, settle time.Duration) {
	t.Helper()

	deadline := time.Now().Add(settle)

	for runtime.NumGoroutine() > before {
		if time.Now().After(deadline) {
			t.Fatalf("goroutine leak: before=%d after=%d", before, runtime.NumGoroutine())

			return // Fatalf does not return on a real *testing.T; this guards fakes.
		}

		time.Sleep(settlePollInterval)
	}
}

// Delay derives a small delay from the seed. It is zero when bit 0 of mask is
// unset (sync mode), otherwise between 1 and 10 units of 200µs. Negative seeds
// are folded to positive ones.
func Delay(seed int64, mask uint8) time.Duration {
	if mask&delayModeAsync == 0 {
		return 0
	}

	if seed < 0 {
		seed = -seed
	}

	// math.MinInt64 stays negative after negation, so the modulo can be negative:
	// fold it back into [0, delayMaxSteps).
	step := seed % delayMaxSteps
	if step < 0 {
		step += delayMaxSteps
	}

	return time.Duration(step+1) * delayUnit
}

// FailingWriter is an io.Writer that succeeds for the first FailAt calls, then
// fails every call with ErrWrite. It is safe for concurrent use.
type FailingWriter struct {
	// FailAt is the number of successful writes before failures start.
	FailAt int

	calls          int64
	callsAfterFail int64
}

// Write implements io.Writer.
func (w *FailingWriter) Write(p []byte) (int, error) {
	n := int(atomic.AddInt64(&w.calls, 1))
	if n > w.FailAt+1 {
		atomic.AddInt64(&w.callsAfterFail, 1)
	}

	if n > w.FailAt {
		return 0, ErrWrite
	}

	return len(p), nil
}

// Calls returns the number of Write calls so far.
func (w *FailingWriter) Calls() int64 { return atomic.LoadInt64(&w.calls) }

// CallsAfterFail returns the number of Write calls made after the first failed one.
func (w *FailingWriter) CallsAfterFail() int64 { return atomic.LoadInt64(&w.callsAfterFail) }
