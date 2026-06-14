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

package xqueue

// Queue is a FIFO queue backed by a growable ring buffer. Popped slots are
// zeroed so that dequeued values are not pinned in memory, and the backing
// array is reused in place: a balanced push/pop workload does not allocate in
// steady state, and a sustained imbalance (one producer permanently ahead of
// the consumer) keeps the buffer bounded to the peak depth rather than letting
// a dead prefix grow without bound.
//
// Queue is not safe for concurrent use.
type Queue[T any] struct {
	items []T
	head  int
	tail  int
	count int
}

// Push appends a value to the back of the queue.
func (q *Queue[T]) Push(value T) {
	if q.count == len(q.items) {
		q.grow()
	}

	q.items[q.tail] = value

	q.tail++
	if q.tail == len(q.items) {
		q.tail = 0
	}

	q.count++
}

// Pop removes and returns the value at the front of the queue.
// It panics if the queue is empty.
func (q *Queue[T]) Pop() T {
	if q.count == 0 {
		panic("xqueue: Pop from empty queue")
	}

	value := q.items[q.head]

	var zero T
	q.items[q.head] = zero

	q.head++
	if q.head == len(q.items) {
		q.head = 0
	}

	q.count--

	return value
}

// Len returns the number of values in the queue.
func (q *Queue[T]) Len() int {
	return q.count
}

// Reset empties the queue and releases the backing array.
func (q *Queue[T]) Reset() {
	q.items = nil
	q.head = 0
	q.tail = 0
	q.count = 0
}

// grow doubles the backing array and re-linearizes the live region so that it
// starts at index 0.
func (q *Queue[T]) grow() {
	newCap := len(q.items) * 2
	if newCap == 0 {
		newCap = 4
	}

	buf := make([]T, newCap)

	// Copy the live region in order, starting from head and wrapping around.
	n := copy(buf, q.items[q.head:])
	copy(buf[n:], q.items[:q.tail])

	q.items = buf
	q.head = 0
	q.tail = q.count
}
