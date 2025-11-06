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

package ro

import (
	"fmt"
	"runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
	"go.uber.org/goleak"
)

// https://github.com/stretchr/testify/issues/1101
func testWithTimeout(t *testing.T, timeout time.Duration) {
	t.Helper()

	testFinished := make(chan struct{})

	t.Cleanup(func() {
		close(testFinished)
	})

	line := ""
	funcName := ""

	var pc [1]uintptr
	n := runtime.Callers(2, pc[:])
	if n > 0 {
		frames := runtime.CallersFrames(pc[:])
		frame, _ := frames.Next()
		line = frame.File + ":" + strconv.Itoa(frame.Line)
		funcName = frame.Function
	}

	go func() {
		select {
		case <-testFinished:
		case <-time.After(timeout):
			if line == "" || funcName == "" {
				panic(fmt.Sprintf("Test timed out after: %v", timeout))
			}
			panic(fmt.Sprintf("%s: Test timed out after: %v\n%s", funcName, timeout, line))

			// t.Errorf("Test timed out after: %v", timeout)
			// os.Exit(1)
		}
	}()
}

func passThrough[T any]() func(Observable[T]) Observable[T] {
	return func(observable Observable[T]) Observable[T] {
		return observable
	}
}

func syncMapLength(m *sync.Map) int {
	size := 0

	m.Range(func(key, value any) bool {
		size++
		return true
	})

	return size
}

func t2ToSliceB[A, B any](slice []lo.Tuple2[A, B]) []B {
	return lo.Map(slice, func(t lo.Tuple2[A, B], _ int) B {
		return t.B
	})
}

func TestMain(m *testing.M) {
	goleak.VerifyTestMain(m)
}

func TestKind_String(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.Equal("Next", KindNext.String())
	is.Equal("Error", KindError.String())
	is.Equal("Complete", KindComplete.String())

	is.PanicsWithValue("you shall not pass", func() {
		_ = Kind(42).String()
	})
}

func TestNotification(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.Equal(Notification[int]{KindNext, 42, nil}, NewNotificationNext(42))
	is.Equal(Notification[int]{KindError, 0, assert.AnError}, NewNotificationError[int](assert.AnError))
	is.Equal(Notification[int]{KindComplete, 0, nil}, NewNotificationComplete[int]())
}

func TestNotification_String(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	is.Equal("Next(42)", NewNotificationNext(42).String())
	is.Equal("Error(assert.AnError general error for testing)", NewNotificationError[int](assert.AnError).String())
	is.Equal("Complete()", NewNotificationComplete[int]().String())

	is.Equal("Error(nil)", Notification[int]{KindError, 0, nil}.String())
	is.PanicsWithValue("you shall not pass", func() {
		n := Notification[int]{Kind(42), 0, nil}
		_ = n.String()
	})
}

func TestProcessNotification(t *testing.T) {
	t.Parallel()
	is := assert.New(t)
	{
		var value int

		processNotification(
			NewNotificationNext(42),
			func(v int) { value = v },
			func(err error) { is.Fail("should not enter here") },
			func() { is.Fail("should not enter here") },
		)
		is.Equal(42, value)
	}

	{
		var value error

		processNotification(
			NewNotificationError[int](assert.AnError),
			func(v int) { is.Fail("should not enter here") },
			func(err error) { value = err },
			func() { is.Fail("should not enter here") },
		)
		is.Equal(assert.AnError, value)
	}

	{
		var value int

		processNotification(
			NewNotificationComplete[int](),
			func(v int) { is.Fail("should not enter here") },
			func(err error) { is.Fail("should not enter here") },
			func() { value = 42 },
		)
		is.Equal(42, value)
	}

	{
		is.PanicsWithValue("you shall not pass", func() {
			processNotification(
				Notification[int]{Kind(42), 0, nil},
				func(v int) { is.Fail("should not enter here") },
				func(err error) { is.Fail("should not enter here") },
				func() { is.Fail("should not enter here") },
			)
		})
	}
}

func TestProcessNotificationWithObserver(t *testing.T) {
	t.Parallel()
	is := assert.New(t)
	{
		var value int

		processNotificationWithObserver(
			NewNotificationNext(42),
			NewObserver(
				func(v int) { value = v },
				func(err error) { is.Fail("should not enter here") },
				func() { is.Fail("should not enter here") },
			),
		)
		is.Equal(42, value)
	}

	{
		var value error

		processNotificationWithObserver(
			NewNotificationError[int](assert.AnError),
			NewObserver(
				func(v int) { is.Fail("should not enter here") },
				func(err error) { value = err },
				func() { is.Fail("should not enter here") },
			),
		)
		is.Equal(assert.AnError, value)
	}

	{
		var value int

		processNotificationWithObserver(
			NewNotificationComplete[int](),
			NewObserver(
				func(v int) { is.Fail("should not enter here") },
				func(err error) { is.Fail("should not enter here") },
				func() { value = 42 },
			),
		)
		is.Equal(42, value)
	}

	{
		is.PanicsWithValue("you shall not pass", func() {
			processNotificationWithObserver(
				Notification[int]{Kind(42), 0, nil},
				NewObserver(
					func(v int) { is.Fail("should not enter here") },
					func(err error) { is.Fail("should not enter here") },
					func() { is.Fail("should not enter here") },
				),
			)
		})
	}
}
