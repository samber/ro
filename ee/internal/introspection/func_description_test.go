// Copyright 2025 samber.
//
// Licensed as an Enterprise License (the "License"); you may not use
// this file except in compliance with the License. You may obtain
// a copy of the License at:
//
// https://github.com/samber/ro/blob/main/licenses/LICENSE.ee.md
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


package introspection

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

type fake struct{}

var (
	packageName     = reflect.TypeOf(fake{}).PkgPath()
	currentFileName = fmt.Sprintf("%s/func_description_test.go", packageName)
)

func TestGetFunctionDescription_args(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// one-liner + external function
	description, err := pipe2(ro.Just(1, 2, 3), ro.Map(func(x int) int { return x * 2 }), ro.Take[int](2))
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 24, 22))
	is.Len(description.Arguments, 2)
	is.Equal("ro.Map(...)", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 24, 46))
	is.Equal("ro.Take[int](...)", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 24, 88))
	is.Nil(err)

	// multi-line + same-package function
	description, err = pipe2(
		ro.Of(1, 2, 3),
		ro.Map(func(x int) int {
			return x * 2
		}),
		distinct[int](),
	)
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 35, 21))
	is.Len(description.Arguments, 2)
	is.Equal("ro.Map(...)", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 37, 3))
	is.Equal("distinct[int](...)", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 40, 3))
	is.Nil(err)

	// multi-line + local function + literal function
	aLocalOperator := func(o ro.Observable[int]) ro.Observable[int] {
		return o
	}
	description, err = pipe2(
		ro.Of(1, 2, 3),
		aLocalOperator,
		func(o ro.Observable[int]) ro.Observable[int] {
			return o
		},
	)
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 55, 21))
	is.Len(description.Arguments, 2)
	is.Equal("aLocalOperator", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 57, 3))
	is.Equal("func(...) { ... }", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 58, 3))
	is.Nil(err)

	// multi-line + local functions
	source := ro.Just(1, 2, 3)
	OPERATOR1 := ro.Map(func(x int) int {
		return x * 2
	})
	OPERATOR2 := ro.Take[int](2)
	description, err = pipe2(
		source,
		OPERATOR1,
		OPERATOR2,
	)
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 77, 21))
	is.Len(description.Arguments, 2)
	is.Equal("OPERATOR1", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 79, 3))
	is.Equal("OPERATOR2", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 80, 3))
	is.Nil(err)

	// multi-line + local functions + parentheses
	description, err = pipe2(
		ro.Just(1, 2, 3),
		(OPERATOR1),
		(OPERATOR2),
	)
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 92, 21))
	is.Len(description.Arguments, 2)
	is.Equal("OPERATOR1", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 94, 3))
	is.Equal("OPERATOR2", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 95, 3))
	is.Nil(err)

	// multi-line + local functions selector + struct/map index selector
	myStructs := []struct {
		Fun func(ro.Observable[int]) ro.Observable[int]
	}{
		{
			Fun: ro.Map(func(x int) int {
				return x * 2
			}),
		},
	}
	myMap := map[string]func(func(int) int) func(ro.Observable[int]) ro.Observable[int]{
		"map": ro.Map[int, int],
	}
	description, err = pipe2(
		ro.Just(1, 2, 3),
		myStructs[0].Fun,
		myMap["map"]((func(x int) int {
			return x * 2
		})),
	)
	is.Equal("pipe2", description.Name)
	is.Contains(description.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 119, 21))
	is.Len(description.Arguments, 2)
	is.Equal("myStructs[0].Fun", description.Arguments[0].Name)
	is.Contains(description.Arguments[0].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 121, 3))
	is.Equal("myMap[\"map\"](...)", description.Arguments[1].Name)
	is.Contains(description.Arguments[1].Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 122, 3))
	is.Nil(err)
}

func TestGetFunctionDescription_skipCaller(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	// skip caller
	desc, err := GetFunctionDescription(-1, 0)
	is.Equal("GetFunctionDescription", desc.Name)
	is.Contains(desc.Pos, fmt.Sprintf("%s:%d:%d", currentFileName, 141, 15))
	is.Nil(err)

	// skip caller overflow (positive)
	desc, err = GetFunctionDescription(100, 0)
	is.Nil(desc)
	is.NotNil(err)
	is.EqualError(err, "failed to collect caller description")

	// skip caller overflow (negative)
	desc, err = GetFunctionDescription(-100, 0)
	is.NotNil(desc)
	is.Nil(err)
}

func TestGetFunctionDescription_skipArg(t *testing.T) {
	t.Parallel()

	t.Run("don't skip", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		desc, err := GetFunctionDescription(-1, 0)
		is.Len(desc.Arguments, 2)
		is.Nil(err)
	})

	t.Run("skip 1 argument", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		desc, err := GetFunctionDescription(-1, 1)
		is.Len(desc.Arguments, 1)
		is.Nil(err)
	})

	t.Run("skip argument overflow (positive)", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		desc, err := GetFunctionDescription(-1, 100)
		is.Len(desc.Arguments, 0)
		is.Nil(err)
	})

	t.Run("skip argument overflow (negative)", func(t *testing.T) {
		t.Parallel()
		is := assert.New(t)

		desc, err := GetFunctionDescription(-1, -100)
		is.Len(desc.Arguments, 2)
		is.Nil(err)
	})
}
