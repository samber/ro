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

package fuzz

import (
	"testing"

	"github.com/samber/ro"
)

const (
	// innerValueStride separates the values of two inner sources: inner i emits i*stride, i*stride+1, ...
	// so a value tells which inner emitted it.
	innerValueStride = 1000

	// maxInnerItems is the largest inner source. Small inners make many of them finish while others start.
	maxInnerItems = 4

	// maxConcatSources bounds the arity of ConcatWith, which takes its sources as arguments.
	maxConcatSources = 8

	// maxItemsBeforeStop bounds how many values a target lets through before it stops the stream.
	maxItemsBeforeStop = 12
)

// higherOrderSources is the input of the operators that merge or concatenate inner sources: `count`
// inner sources, and an outer source whose item i selects inner i.
//
// Inner i has (firstInnerSize+i) % (maxInnerItems+1) items, so sizes cycle through 0..maxInnerItems,
// empty inners included. Even and odd inners are synchronous or asynchronous independently.
type higherOrderSources struct {
	outerCounter subscriptionCounter
	innerCounter subscriptionCounter

	outer  *source
	inners []ro.Observable[int]

	// emittable lists every value the inners can emit, in inner order.
	emittable []int
}

func newHigherOrderSources(count int, firstInnerSize uint8, asyncOuter, asyncEvenInners, asyncOddInners bool) *higherOrderSources {
	sources := &higherOrderSources{outer: newSource(count, asyncOuter)}

	for index := 0; index < count; index++ {
		size := (int(firstInnerSize) + index) % (maxInnerItems + 1)
		async := asyncOddInners

		if index%2 == 0 {
			async = asyncEvenInners
		}

		inner := newSource(size, async).startingAt(index * innerValueStride).yieldingWith(int64(index))
		sources.inners = append(sources.inners, countSubscriptions(&sources.innerCounter, inner.observable()))

		for item := 0; item < size; item++ {
			sources.emittable = append(sources.emittable, index*innerValueStride+item)
		}
	}

	return sources
}

// outerItems is the counted outer source: 0..count-1.
func (s *higherOrderSources) outerItems() ro.Observable[int] {
	return countSubscriptions(&s.outerCounter, s.outer.observable())
}

// project selects the inner source of an outer item.
func (s *higherOrderSources) project(outerItem int) ro.Observable[int] { return s.inners[outerItem] }

// upstreams are the counters that must read zero once the stream ended.
func (s *higherOrderSources) upstreams() []*subscriptionCounter {
	return []*subscriptionCounter{&s.innerCounter, &s.outerCounter}
}

// expectOnlyEmittedValues checks that nothing was invented or duplicated. When ordered, it also checks that
// the values are the first values of the inners, one inner after the other.
func (s *higherOrderSources) expectOnlyEmittedValues(t *testing.T, got *recorder[int], ordered bool) {
	t.Helper()

	values := got.received()
	if len(values) > len(s.emittable) {
		t.Fatalf("%d values received, only %d can exist", len(values), len(s.emittable))
	}

	if !ordered {
		return
	}

	for index, value := range values {
		if value != s.emittable[index] {
			t.Fatalf("concatenation order broken at %d: got %d, want %d", index, value, s.emittable[index])
		}
	}
}

// expectEveryValue checks that every emittable value arrived and that the stream completed once.
func (s *higherOrderSources) expectEveryValue(t *testing.T, got *recorder[int], ordered bool) {
	t.Helper()

	s.expectOnlyEmittedValues(t, got, ordered)

	if received := got.valueCount(); received != len(s.emittable) {
		t.Fatalf("lost values: got %d of %d", received, len(s.emittable))
	}

	got.expectCompletedOnce(t)
}

// expectTakenValues checks that Take(limit) delivered the first values and no other.
func (s *higherOrderSources) expectTakenValues(t *testing.T, got *recorder[int], limit int, ordered bool) {
	t.Helper()

	s.expectOnlyEmittedValues(t, got, ordered)

	if received, want := got.valueCount(), smaller(limit, len(s.emittable)); received != want {
		t.Fatalf("take(%d) delivered %d of %d available values", limit, received, len(s.emittable))
	}
}
