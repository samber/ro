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

package xsync_test

import (
	"sync"
	"testing"
	"time"

	"github.com/samber/ro/internal/xsync"
)

func TestRWMutexWithLock(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	m := xsync.NewRWMutexWithLock()

	// Test Lock/Unlock
	m.Lock()

	locked := make(chan bool)
	unlocked := make(chan struct{})

	go func() {
		// Try to acquire the lock - should block until main goroutine unlocks
		m.Lock()

		locked <- true

		m.Unlock() // Unlock after acquiring to free the mutex
		close(unlocked)
	}()

	// Wait a short time to ensure the goroutine is blocked
	time.Sleep(5 * time.Millisecond)

	// Check that the goroutine is still blocked
	select {
	case <-locked:
		t.Error("Lock acquired while it should be held")
	default:
		// Expected: goroutine is blocked
	}

	m.Unlock()

	// Now the goroutine should acquire the lock
	select {
	case <-locked:
		// Expected: lock acquired after unlock
	case <-time.After(100 * time.Millisecond):
		t.Error("Lock not acquired after unlock")
	}

	// Wait for the goroutine to finish unlocking
	<-unlocked

	// Test TryLock on unlocked mutex
	if !m.TryLock() {
		t.Error("TryLock failed on unlocked mutex")
	}

	m.Unlock()

	// Test TryLock on locked mutex
	m.Lock()

	if m.TryLock() {
		t.Error("TryLock succeeded on locked mutex")
	}

	m.Unlock()

	// Test RLock/RUnlock
	m.RLock()

	var wg sync.WaitGroup

	wg.Add(1)

	go func() {
		defer wg.Done()

		m.RLock()   // Should not block
		m.RUnlock() //nolint:staticcheck
	}()

	wg.Wait()
	m.RUnlock()

	// Test TryRLock
	if !m.TryRLock() {
		t.Error("TryRLock failed on unlocked mutex")
	}

	m.RUnlock()

	m.RLock()

	if !m.TryRLock() {
		t.Error("TryRLock failed on RLocked mutex")
	}

	m.RUnlock()
	m.RUnlock()

	m.Lock()

	if m.TryRLock() {
		t.Error("TryRLock succeeded on locked mutex")
	}

	m.Unlock()
}

func TestRWMutexWithoutLock(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	m := xsync.NewRWMutexWithoutLock()

	// Test TryLock
	if !m.TryLock() {
		t.Error("TryLock should always return true")
	}

	// Test Lock/Unlock (should not block)
	done := make(chan bool)

	go func() {
		m.Lock()
		m.Unlock() //nolint:staticcheck

		done <- true
	}()

	select {
	case <-done:
		// Expected: should complete immediately
	case <-time.After(10 * time.Millisecond):
		t.Error("Lock/Unlock should not block")
	}

	// Test TryRLock
	if !m.TryRLock() {
		t.Error("TryRLock should always return true")
	}

	// Test RLock/RUnlock (should not block)
	done = make(chan bool)

	go func() {
		m.RLock()
		m.RUnlock() //nolint:staticcheck

		done <- true
	}()

	select {
	case <-done:
		// Expected: should complete immediately
	case <-time.After(10 * time.Millisecond):
		t.Error("RLock/RUnlock should not block")
	}
}
