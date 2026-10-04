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
	"runtime"
	"sync"
	"sync/atomic"
)

// Mutex is a mutex interface.
type Mutex interface {
	TryLock() bool
	Lock()
	Unlock()
}

/************************
 *    Standard mutex    *
 ************************/

var _ Mutex = (*MutexWithLock)(nil)

// NewMutexWithLock creates a new mutex with a standard mutex.
func NewMutexWithLock() *MutexWithLock {
	return &MutexWithLock{
		mu: sync.Mutex{},
	}
}

// MutexWithLock is a mutex with a standard mutex.
type MutexWithLock struct {
	mu sync.Mutex
}

// TryLock tries to lock the mutex.
func (m *MutexWithLock) TryLock() bool {
	return m.mu.TryLock()
}

// Lock locks the mutex.
func (m *MutexWithLock) Lock() {
	m.mu.Lock()
}

// Unlock unlocks the mutex.
func (m *MutexWithLock) Unlock() {
	m.mu.Unlock()
}

/************************
 *    Fast mutex        *
 ************************/

var _ Mutex = (*MutexWithSpinlock)(nil)

// NewMutexWithSpinlock creates a new mutex with a spinlock.
// It is faster than the standard mutex, but it is CPU-intensive.
func NewMutexWithSpinlock() *MutexWithSpinlock {
	return &MutexWithSpinlock{
		lock: 0,
	}
}

// MutexWithSpinlock is a mutex with a spinlock.
type MutexWithSpinlock struct {
	lock int32 // 0 or 1
}

// TryLock tries to lock the mutex. It returns true if the lock was acquired.
func (m *MutexWithSpinlock) TryLock() bool {
	return atomic.CompareAndSwapInt32(&m.lock, 0, 1)
}

// Lock locks the mutex.
func (m *MutexWithSpinlock) Lock() {
	for !atomic.CompareAndSwapInt32(&m.lock, 0, 1) {
		runtime.Gosched()
	}
}

// Unlock unlocks the mutex.
func (m *MutexWithSpinlock) Unlock() {
	atomic.StoreInt32(&m.lock, 0)
}

/************************
 *      Fake mutex      *
 ************************/

var _ Mutex = (*MutexWithoutLock)(nil)

// NewMutexWithoutLock creates a new mutex without a lock.
func NewMutexWithoutLock() *MutexWithoutLock {
	return &MutexWithoutLock{}
}

// MutexWithoutLock is a mutex without a lock.
type MutexWithoutLock struct{}

// TryLock always returns true.
func (m *MutexWithoutLock) TryLock() bool {
	return true
}

// Lock does nothing.
func (m *MutexWithoutLock) Lock() {
}

// Unlock does nothing.
func (m *MutexWithoutLock) Unlock() {
}
