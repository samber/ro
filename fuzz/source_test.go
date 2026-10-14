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
	"context"
	"runtime"
	"sync"
	"time"

	"github.com/samber/ro"
)

// source is a test Observable that emits firstValue, firstValue+1... and then ends. Build it with newSource and
// refine it with the chained methods.
//
// A synchronous source emits inside Subscribe: the teardown of the subscriber exists only after
// the last item. An asynchronous source emits from its own goroutine: its Next races Unsubscribe and
// the terminal notification. Every target must run against both kinds.
type source struct {
	items      int
	async      bool
	ignoreStop bool
	failure    error
	neverEnds  bool
	firstValue int
	yieldSeed  int64

	// producers tracks the goroutines of an asynchronous source.
	producers sync.WaitGroup
}

// newSource emits 0..items-1 and then completes. It stops as soon as the downstream is closed
// (synchronous source) or the teardown ran (asynchronous source).
func newSource(items int, async bool) *source {
	return &source{items: items, async: async}
}

// ignoringStop makes the source emit every item even after the downstream is closed, like a source
// that cannot be interrupted. Operators must not evaluate user callbacks for those late items.
func (s *source) ignoringStop(ignore bool) *source {
	s.ignoreStop = ignore

	return s
}

// failingAtEnd makes the source end with err instead of completing.
func (s *source) failingAtEnd(err error) *source {
	s.failure = err

	return s
}

// neverEnding makes the source emit its items and then stay subscribed, without any terminal notification.
func (s *source) neverEnding() *source {
	s.neverEnds = true

	return s
}

// startingAt shifts the emitted values, so that several sources can be told apart: they emit
// firstValue..firstValue+items-1.
func (s *source) startingAt(firstValue int) *source {
	s.firstValue = firstValue

	return s
}

// yieldingWith makes the producer goroutine of an asynchronous source yield the processor at
// positions chosen by seed, so that different seeds explore different interleavings.
func (s *source) yieldingWith(seed int64) *source {
	s.yieldSeed = seed

	return s
}

// waitForProducers blocks until the goroutines of an asynchronous source returned. It must run under
// runWithinDeadline.
func (s *source) waitForProducers() { s.producers.Wait() }

func (s *source) observable() ro.Observable[int] {
	if s.async {
		return ro.NewUnsafeObservableWithContext(s.emitFromGoroutine)
	}

	return ro.NewUnsafeObservableWithContext(s.emitInline)
}

func (s *source) emitInline(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
	for i := 0; i < s.items; i++ {
		if !s.ignoreStop && destination.IsClosed() {
			return nil
		}

		destination.NextWithContext(ctx, s.firstValue+i)
	}

	s.end(ctx, destination)

	return nil
}

func (s *source) emitFromGoroutine(ctx context.Context, destination ro.Observer[int]) ro.Teardown {
	stopped := make(chan struct{})

	var stop sync.Once

	s.producers.Add(1)

	go func() {
		defer s.producers.Done()

		for i := 0; i < s.items; i++ {
			if !s.ignoreStop && isClosed(stopped) {
				return
			}

			yieldAt(s.yieldSeed, i)
			destination.NextWithContext(ctx, s.firstValue+i)
		}

		s.end(ctx, destination)
	}()

	// The teardown never waits for the goroutine: it only signals it.
	return func() { stop.Do(func() { close(stopped) }) }
}

func (s *source) end(ctx context.Context, destination ro.Observer[int]) {
	switch {
	case s.neverEnds:
	case s.failure != nil:
		destination.ErrorWithContext(ctx, s.failure)
	default:
		destination.CompleteWithContext(ctx)
	}
}

func isClosed(channel <-chan struct{}) bool {
	select {
	case <-channel:
		return true
	default:
		return false
	}
}

// yieldAt yields the processor at positions chosen by seed. step identifies the position in the sequence.
func yieldAt(seed int64, step int) {
	// Mixing seed and step keeps neighbouring steps from yielding together.
	mixed := uint64(seed)*6364136223846793005 + uint64(step)*1442695040888963407

	// The 3 top bits give 0..7: yield on 2 values, sleep on 1, run on 5.
	switch mixed >> 61 {
	case 0, 1:
		runtime.Gosched()
	case 2:
		time.Sleep(time.Microsecond)
	}
}

// sequence returns from..to-1: the values a source emits before ending.
func sequence(from, to int) []int {
	values := []int{}
	for i := from; i < to; i++ {
		values = append(values, i)
	}

	return values
}

func smaller(a, b int) int {
	if a < b {
		return a
	}

	return b
}

// itemAt returns the value a newSource(count, ...) emits at index, or nothing when index is out of range.
func itemAt(index, count int) []int {
	if index < count {
		return []int{index}
	}

	return []int{}
}
