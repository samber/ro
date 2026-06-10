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

// Queue is a FIFO queue backed by a slice. Popped slots are zeroed so that
// dequeued values are not pinned in memory, and the backing array is reused
// once the queue is drained, so a balanced push/pop workload does not
// allocate in steady state.
//
// Queue is not safe for concurrent use.
type Queue[T any] struct {
	items []T
	head  int
}

// Push appends a value to the back of the queue.
func (q *Queue[T]) Push(value T) {
	q.items = append(q.items, value)
}

// Pop removes and returns the value at the front of the queue.
// It panics if the queue is empty.
func (q *Queue[T]) Pop() T {
	value := q.items[q.head]

	var zero T
	q.items[q.head] = zero
	q.head++

	if q.head == len(q.items) {
		q.items = q.items[:0]
		q.head = 0
	}

	return value
}

// Len returns the number of values in the queue.
func (q *Queue[T]) Len() int {
	return len(q.items) - q.head
}

// Reset empties the queue and releases the backing array.
func (q *Queue[T]) Reset() {
	q.items = nil
	q.head = 0
}
