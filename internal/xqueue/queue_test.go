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

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestQueuePushPop(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := NewQueue[int]()

	is.Equal(0, q.Len())

	q.Push(1)
	q.Push(2)
	q.Push(3)
	is.Equal(3, q.Len())

	is.Equal(1, q.Pop())
	is.Equal(2, q.Pop())
	is.Equal(1, q.Len())

	q.Push(4)
	is.Equal(3, q.Pop())
	is.Equal(4, q.Pop())
	is.Equal(0, q.Len())
}

func TestQueueReusesCapacityWhenDrained(t *testing.T) {
	// Not parallel: testing.AllocsPerRun is unreliable when other tests run
	// concurrently, since its result depends on whole-program GC behavior.
	is := assert.New(t)

	q := NewQueue[int]()

	// Warm up the backing array.
	q.Push(1)
	q.Pop()

	allocs := testing.AllocsPerRun(100, func() {
		q.Push(1)
		q.Pop()
	})
	is.Equal(0.0, allocs)
}

func TestQueueBoundedUnderSustainedSkew(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// One producer permanently ahead of the consumer: the live depth stays
	// constant but head keeps advancing. The backing array must stay bounded
	// to the peak depth rather than growing without bound.
	q := &queueImpl[int]{}
	for i := 0; i < 8; i++ {
		q.Push(i)
	}

	for i := 0; i < 100_000; i++ {
		q.Push(i)
		_ = q.Pop()
	}

	is.Equal(8, q.Len())
	is.LessOrEqual(cap(q.items), 16)
}

func TestQueueShrinksAfterBurst(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := &queueImpl[int]{}

	// Grow the backing array with a burst.
	for i := 0; i < 256; i++ {
		q.Push(i)
	}
	is.GreaterOrEqual(cap(q.items), 256)

	// Drain back down to a single live value: the backing array must be
	// released rather than held at the peak. A small floor is acceptable.
	for q.Len() > 1 {
		_ = q.Pop()
	}
	is.Equal(1, q.Len())
	is.LessOrEqual(cap(q.items), minShrinkCap)

	// The retained value is still correct and in order.
	is.Equal(255, q.Pop())
	is.Equal(0, q.Len())
}

func TestQueuePopEmptyPanics(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := NewQueue[int]()

	is.Panics(func() {
		q.Pop()
	})
}

func TestQueueZeroesPoppedSlots(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := &queueImpl[*int]{}

	v1 := 1
	v2 := 2
	q.Push(&v1)
	q.Push(&v2)
	is.Equal(&v1, q.Pop())

	// The popped slot must not pin the value.
	is.Nil(q.items[0])
	is.Equal(&v2, q.Pop())
}

func TestQueueReset(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := &queueImpl[int]{}

	q.Push(1)
	q.Push(2)
	q.Pop()
	q.Reset()

	is.Equal(0, q.Len())
	is.Nil(q.items)
	is.Equal(0, q.head)

	q.Push(3)
	is.Equal(3, q.Pop())
}

func TestQueueInterleaved(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	q := NewQueue[int]()

	next := 0
	expected := 0

	for i := 0; i < 1000; i++ {
		q.Push(next)
		next++

		if i%3 == 0 {
			is.Equal(expected, q.Pop())
			expected++
		}
	}

	for q.Len() > 0 {
		is.Equal(expected, q.Pop())
		expected++
	}

	is.Equal(next, expected)
}
