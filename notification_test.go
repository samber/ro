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
	"testing"

	"github.com/stretchr/testify/assert"
)

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
