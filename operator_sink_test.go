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
	"strconv"
	"testing"
	"time"

	"github.com/samber/lo"
	"github.com/stretchr/testify/assert"
)

func TestOperatorSinkToSlice(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ToSlice[int]()(Just(1, 2, 3)),
	)
	is.Equal([][]int{{1, 2, 3}}, values)
	is.NoError(err)

	values, err = Collect(
		ToSlice[int]()(Empty[int]()),
	)
	is.Equal([][]int{{}}, values)
	is.NoError(err)

	values, err = Collect(
		ToSlice[int]()(Throw[int](assert.AnError)),
	)
	is.Equal([][]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorSinkToMap(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	mapper := func(v int) (string, string) {
		return strconv.FormatInt(int64(v), 10), strconv.FormatInt(int64(v), 10)
	}

	values, err := Collect(
		ToMap(mapper)(Just(1, 2, 3)),
	)
	is.Equal([]map[string]string{{"1": "1", "2": "2", "3": "3"}}, values)
	is.NoError(err)

	values, err = Collect(
		ToMap(mapper)(Empty[int]()),
	)
	is.Equal([]map[string]string{{}}, values)
	is.NoError(err)

	values, err = Collect(
		ToMap(mapper)(Throw[int](assert.AnError)),
	)
	is.Equal([]map[string]string{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorSinkToMapI(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	mapper := func(v int, _ int64) (string, string) {
		return strconv.FormatInt(int64(v), 10), strconv.FormatInt(int64(v), 10)
	}

	values, err := Collect(
		ToMapI(mapper)(Just(1, 2, 3)),
	)
	is.Equal([]map[string]string{{"1": "1", "2": "2", "3": "3"}}, values)
	is.NoError(err)

	values, err = Collect(
		ToMapI(func(v int, i int64) (string, string) {
			is.Equal(int(i), v)
			return strconv.FormatInt(int64(v), 10), strconv.FormatInt(int64(v), 10)
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]map[string]string{{"0": "0", "1": "1", "2": "2", "3": "3"}}, values)
	is.NoError(err)

	values, err = Collect(
		ToMapI(mapper)(Empty[int]()),
	)
	is.Equal([]map[string]string{{}}, values)
	is.NoError(err)

	values, err = Collect(
		ToMapI(mapper)(Throw[int](assert.AnError)),
	)
	is.Equal([]map[string]string{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorSinkToChannel(t *testing.T) {
	t.Parallel()
	testWithTimeout(t, 100*time.Millisecond)
	is := assert.New(t)

	values, err := Collect(
		ToChannel[int](10)(Just(1, 2, 3)),
	)
	is.Len(values, 1)
	all := lo.ChannelToSlice(values[0])
	is.Equal([]Notification[int]{
		NewNotificationNext(1),
		NewNotificationNext(2),
		NewNotificationNext(3),
		NewNotificationComplete[int](),
	}, all)
	is.NoError(err)

	values, err = Collect(
		ToChannel[int](10)(Empty[int]()),
	)
	is.Len(values, 1)
	all = lo.ChannelToSlice(values[0])
	is.Equal([]Notification[int]{
		NewNotificationComplete[int](),
	}, all)
	is.NoError(err)

	values, err = Collect(
		ToChannel[int](10)(Throw[int](assert.AnError)),
	)
	is.Len(values, 1)
	all = lo.ChannelToSlice(values[0])
	is.Equal([]Notification[int]{
		NewNotificationError[int](assert.AnError),
	}, all)
	is.NoError(err)
}
