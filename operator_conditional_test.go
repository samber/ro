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

func TestOperatorConditionalAll(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int) bool {
		return v%2 == 0
	}

	values, err := Collect(
		All(odd)(Just(1, 2, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		All(odd)(Just(1, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		All(odd)(Just(2, 4)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		All(odd)(Empty[int]()),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		All(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalAllI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int, _ int64) bool {
		return v%2 == 0
	}

	values, err := Collect(
		AllI(odd)(Just(1, 2, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		AllI(func(v int, index int64) bool {
			is.Equal(v, int(index))
			return v%2 == 0
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		AllI(odd)(Just(1, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		AllI(odd)(Just(2, 4)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		AllI(odd)(Empty[int]()),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		AllI(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalContains(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int) bool {
		return v%2 == 0
	}

	values, err := Collect(
		Contains(odd)(Just(1, 2, 3)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		Contains(odd)(Just(1, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Contains(odd)(Just(2)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		Contains(odd)(Just(2, 4, 8)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		Contains(odd)(Empty[int]()),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Contains(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalContainsI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int, _ int64) bool {
		return v%2 == 0
	}

	values, err := Collect(
		ContainsI(odd)(Just(1, 2, 3)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(func(v int, i int64) bool {
			is.Equal(v, int(i))
			return v%2 == 0
		})(Just(0, 1, 2, 3)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(odd)(Just(1, 3)),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(odd)(Just(2)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(odd)(Just(2, 4, 8)),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(odd)(Empty[int]()),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		ContainsI(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalFind(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int) bool {
		return v%2 == 0
	}

	values, err := Collect(
		Find(odd)(Just(1, 2, 3)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		Find(odd)(Just(1, 2, 3, 4)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		Find(odd)(Just(1, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Find(odd)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Find(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalFindI(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	odd := func(v int, _ int64) bool {
		return v%2 == 0
	}

	values, err := Collect(
		FindI(odd)(Just(1, 2, 3)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		FindI(odd)(Just(1, 2, 3, 4)),
	)
	is.Equal([]int{2}, values)
	is.NoError(err)

	values, err = Collect(
		FindI(odd)(Just(1, 3)),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		FindI(odd)(Empty[int]()),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		FindI(odd)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalIif(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	tRue := func() bool { return true }
	fAlse := func() bool { return false }

	values, err := Collect(
		Iif(tRue, Just(1, 2, 3), Just(4, 5, 6))(),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		Iif(fAlse, Just(1, 2, 3), Just(4, 5, 6))(),
	)
	is.Equal([]int{4, 5, 6}, values)
	is.NoError(err)

	values, err = Collect(
		Iif(fAlse, Empty[int](), Empty[int]())(),
	)
	is.Equal([]int{}, values)
	is.NoError(err)

	values, err = Collect(
		Iif(fAlse, Just(1, 2, 3), Throw[int](assert.AnError))(),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalDefaultIfEmpty(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		DefaultIfEmpty(42)(Just(1, 2, 3)),
	)
	is.Equal([]int{1, 2, 3}, values)
	is.NoError(err)

	values, err = Collect(
		DefaultIfEmpty(42)(Empty[int]()),
	)
	is.Equal([]int{42}, values)
	is.NoError(err)

	values, err = Collect(
		DefaultIfEmpty(42)(Throw[int](assert.AnError)),
	)
	is.Equal([]int{}, values)
	is.EqualError(err, assert.AnError.Error())
}

func TestOperatorConditionalSequenceEqual(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	values, err := Collect(
		Pipe1(
			Just(1, 2, 3),
			SequenceEqual(Just(1, 2, 3)),
		),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1, 2, 3),
			SequenceEqual(Just(1, 3, 2)),
		),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int](),
			SequenceEqual(Just(1, 2, 3)),
		),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1, 2, 3),
			SequenceEqual(Empty[int]()),
		),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Empty[int](),
			SequenceEqual(Empty[int]()),
		),
	)
	is.Equal([]bool{true}, values)
	is.NoError(err)

	// Different lengths with a common prefix: the shorter sequence is a prefix of the longer one.
	values, err = Collect(
		Pipe1(
			Just(1, 2),
			SequenceEqual(Just(1, 2, 3)),
		),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Just(1, 2, 3),
			SequenceEqual(Just(1, 2)),
		),
	)
	is.Equal([]bool{false}, values)
	is.NoError(err)

	values, err = Collect(
		Pipe1(
			Throw[int](assert.AnError),
			SequenceEqual(Just(1, 2, 3)),
		),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())

	values, err = Collect(
		Pipe1(
			Just(1, 2, 3),
			SequenceEqual(Throw[int](assert.AnError)),
		),
	)
	is.Equal([]bool{}, values)
	is.EqualError(err, assert.AnError.Error())
}
