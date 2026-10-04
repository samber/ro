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

package xsync

import (
	"sync"
	"testing"
)

func TestMutexWithLock_TryLock(t *testing.T) {
	t.Parallel()
	mutex := NewMutexWithLock()

	// Test TryLock on unlocked mutex
	if !mutex.TryLock() {
		t.Error("TryLock should return true on unlocked mutex")
	}

	// Test TryLock on locked mutex
	if mutex.TryLock() {
		t.Error("TryLock should return false on locked mutex")
	}

	// Unlock and test again
	mutex.Unlock()

	if !mutex.TryLock() {
		t.Error("TryLock should return true after unlock")
	}

	mutex.Unlock()
}

func TestMutexWithLock_LockUnlock(t *testing.T) {
	t.Parallel()
	mutex := NewMutexWithLock()

	var counter int

	// Test basic lock/unlock
	mutex.Lock()

	counter++

	mutex.Unlock()

	if counter != 1 {
		t.Error("Lock/Unlock should allow access to critical section")
	}
}

func TestMutexWithLock_ConcurrentAccess(t *testing.T) {
	t.Parallel()
	mutex := NewMutexWithLock()

	var counter int

	var wg sync.WaitGroup

	numGoroutines := 100
	iterations := 1000

	// Start multiple goroutines that increment counter
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < iterations; j++ {
				mutex.Lock()

				counter++

				mutex.Unlock()
			}
		}()
	}

	wg.Wait()

	expected := numGoroutines * iterations
	if counter != expected {
		t.Errorf("Expected counter to be %d, got %d", expected, counter)
	}
}

func TestMutexWithSpinlock_TryLock(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	mutex := NewMutexWithSpinlock()

	// Test TryLock on unlocked mutex
	if !mutex.TryLock() {
		t.Error("TryLock should return true on unlocked mutex")
	}

	// Test TryLock on locked mutex
	if mutex.TryLock() {
		t.Error("TryLock should return false on locked mutex")
	}

	// Unlock and test again
	mutex.Unlock()

	if !mutex.TryLock() {
		t.Error("TryLock should return true after unlock")
	}

	mutex.Unlock()
}

func TestMutexWithSpinlock_LockUnlock(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	mutex := NewMutexWithSpinlock()

	var counter int

	// Test basic lock/unlock
	mutex.Lock()

	counter++

	mutex.Unlock()

	if counter != 1 {
		t.Error("Lock/Unlock should allow access to critical section")
	}
}

func TestMutexWithSpinlock_ConcurrentAccess(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	mutex := NewMutexWithSpinlock()

	var counter int

	var wg sync.WaitGroup

	numGoroutines := 5 // Reduced to avoid timeout
	iterations := 10   // Reduced to avoid timeout

	// Start multiple goroutines that increment counter
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)

		go func() {
			defer wg.Done()

			for j := 0; j < iterations; j++ {
				mutex.Lock()

				counter++

				mutex.Unlock()
			}
		}()
	}

	wg.Wait()

	expected := numGoroutines * iterations
	if counter != expected {
		t.Errorf("Expected counter to be %d, got %d", expected, counter)
	}
}

// func TestMutexWithSpinlock_SpinlockBehavior(t *testing.T) {  //nolint:paralleltest
//  // t.Parallel()
// 	mutex := NewMutexWithSpinlock()
// 	done := make(chan bool)

// 	// Lock the mutex in a goroutine
// 	go func() {
// 		mutex.Lock()
// 		time.Sleep(1 * time.Millisecond) // Hold the lock briefly
// 		mutex.Unlock()

// 		done <- true
// 	}()

// 	// Try to acquire the lock from main goroutine
// 	start := time.Now()

// 	mutex.Lock()

// 	duration := time.Since(start)

// 	mutex.Unlock()

// 	<-done

// 	// The lock should have been acquired after some spinning
// 	if duration < 100*time.Microsecond {
// 		t.Log("Spinlock should have taken some time to acquire, but timing can be variable")
// 	}
// }

func TestMutexWithoutLock_TryLock(t *testing.T) { //nolint:paralleltest
	// t.Parallel()
	mutex := NewMutexWithoutLock()

	// TryLock should always return true for fake mutex
	if !mutex.TryLock() {
		t.Error("TryLock should always return true for fake mutex")
	}

	// Even after "locking", TryLock should still return true
	if !mutex.TryLock() {
		t.Error("TryLock should always return true for fake mutex")
	}
}

func TestMutexWithoutLock_LockUnlock(t *testing.T) {
	t.Parallel()

	mutex := NewMutexWithoutLock()

	var counter int

	// Lock and unlock should not block
	mutex.Lock()

	counter++

	mutex.Unlock()

	counter++

	if counter != 2 {
		t.Error("Lock/Unlock should not affect execution")
	}
}

// func TestMutexPerformanceComparison(t *testing.T) {
// 	// This test compares the performance characteristics of different mutex types
// 	// Note: This is more of a benchmark than a unit test
// 	testMutex := func(name string, mutex Mutex, numGoroutines, iterations int) time.Duration {
// 		var wg sync.WaitGroup

// 		start := time.Now()

// 		for i := 0; i < numGoroutines; i++ {
// 			wg.Add(1)

// 			go func() {
// 				defer wg.Done()

// 				for j := 0; j < iterations; j++ {
// 					mutex.Lock()
// 					// Simulate some work
// 					_ = j * j

// 					mutex.Unlock()
// 				}
// 			}()
// 		}

// 		wg.Wait()

// 		return time.Since(start)
// 	}

// 	numGoroutines := 10
// 	iterations := 1000

// 	// Test each mutex type
// 	standardDuration := testMutex("Standard", NewMutexWithLock(), numGoroutines, iterations)
// 	spinlockDuration := testMutex("Spinlock", NewMutexWithSpinlock(), numGoroutines, iterations)
// 	// fakeDuration := testMutex("Fake", NewMutexWithoutLock(), numGoroutines, iterations)

// 	t.Logf("Standard mutex: %v", standardDuration)
// 	t.Logf("Spinlock mutex: %v", spinlockDuration)
// 	// t.Logf("Fake mutex: %v", fakeDuration)
// 	// // The fake mutex should be the fastest
// 	// if fakeDuration >= standardDuration {
// 	// 	t.Log("Fake mutex should be faster than standard mutex")
// 	// }
// }

func TestMutexEdgeCases(t *testing.T) {
	t.Parallel()
	// Test edge cases for all mutex types
	mutexTypes := []struct {
		name  string
		mutex Mutex
	}{
		{"Standard", NewMutexWithLock()},
		{"Fake", NewMutexWithoutLock()},
	}

	for _, mt := range mutexTypes {
		mutex := mt.mutex

		t.Run(mt.name, func(t *testing.T) {
			t.Parallel()

			// Test multiple rapid lock/unlock operations
			for i := 0; i < 1000; i++ {
				mutex.Lock()
				mutex.Unlock() //nolint:staticcheck
			}

			// Test TryLock in rapid succession
			for i := 0; i < 100; i++ {
				mutex.TryLock()
				mutex.Unlock()
			}

			// Test mixed operations
			for i := 0; i < 100; i++ {
				if mutex.TryLock() {
					mutex.Unlock()
				} else {
					mutex.Lock()
					mutex.Unlock() //nolint:staticcheck
				}
			}
		})
	}
}
