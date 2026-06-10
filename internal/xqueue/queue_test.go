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

	var q Queue[int]

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
	t.Parallel()
	is := assert.New(t)

	var q Queue[int]

	// Warm up the backing array.
	q.Push(1)
	q.Pop()

	allocs := testing.AllocsPerRun(100, func() {
		q.Push(1)
		q.Pop()
	})
	is.Equal(0.0, allocs)
}

func TestQueuePopEmptyPanics(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var q Queue[int]

	is.Panics(func() {
		q.Pop()
	})
}

func TestQueueZeroesPoppedSlots(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	var q Queue[*int]

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

	var q Queue[int]

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

	var q Queue[int]

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
