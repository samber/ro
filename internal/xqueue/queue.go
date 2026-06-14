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

// Queue is a FIFO queue. Popped slots are zeroed so that dequeued values are
// not pinned in memory, and the backing storage is reused in place: a balanced
// push/pop workload does not allocate in steady state, and a sustained
// imbalance (one producer permanently ahead of the consumer) keeps the storage
// bounded to the peak depth rather than letting a dead prefix grow without
// bound.
//
// Queue is not safe for concurrent use.
type Queue[T any] interface {
	// Push appends a value to the back of the queue.
	Push(value T)
	// Pop removes and returns the value at the front of the queue.
	// It panics if the queue is empty.
	Pop() T
	// Len returns the number of values in the queue.
	Len() int
	// Reset empties the queue and releases the backing storage.
	Reset()
}

var _ Queue[int] = (*queueImpl[int])(nil)

// minShrinkCap is the smallest backing array the queue will shrink down to.
// Below it, the savings are not worth the reallocation, and keeping a small
// floor avoids thrashing for queues that hover near empty.
const minShrinkCap = 16

// NewQueue creates a new empty FIFO queue.
func NewQueue[T any]() Queue[T] {
	return &queueImpl[T]{}
}

// queueImpl is a FIFO queue backed by a growable ring buffer.
type queueImpl[T any] struct {
	items []T
	head  int
	tail  int
	count int
}

// Push appends a value to the back of the queue.
func (q *queueImpl[T]) Push(value T) {
	if q.count == len(q.items) {
		newCap := len(q.items) * 2
		if newCap == 0 {
			newCap = 4
		}

		q.resize(newCap)
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
func (q *queueImpl[T]) Pop() T {
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

	// Release backing storage in a single step once a large array has drained
	// down to a handful of items, so a queue that grew for a burst does not
	// hold the peak array after the producer goes quiet. Shrinking only at a
	// low absolute count (rather than progressively while draining) keeps a
	// full drain to one reallocation, and the floor avoids thrashing.
	if q.count <= minShrinkCap && len(q.items) > 2*minShrinkCap {
		q.resize(minShrinkCap)
	}

	return value
}

// Len returns the number of values in the queue.
func (q *queueImpl[T]) Len() int {
	return q.count
}

// Reset empties the queue and releases the backing array.
func (q *queueImpl[T]) Reset() {
	q.items = nil
	q.head = 0
	q.tail = 0
	q.count = 0
}

// resize moves the live region into a freshly allocated array of newCap,
// re-linearizing it so that it starts at index 0. newCap must be >= count.
func (q *queueImpl[T]) resize(newCap int) {
	buf := make([]T, newCap)

	// Copy the live region in order, starting from head and wrapping around.
	n := copy(buf, q.items[q.head:])
	copy(buf[n:], q.items[:q.tail])

	q.items = buf
	q.head = 0
	q.tail = q.count
	if q.tail == newCap {
		q.tail = 0
	}
}
