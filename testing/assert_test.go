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

package testing //nolint:revive

import (
	"testing"

	"github.com/samber/ro"
	"github.com/stretchr/testify/assert"
)

func TestAssert(t *testing.T) {
	t.Parallel()
	is := assert.New(t)

	a, ok := Assert[int](t).(*assertImpl[int])
	is.True(ok)
	is.Equal(t, t)
	is.Empty(a.assertions)
	is.Nil(a.source)

	// complete normally
	Assert[int](t).
		Source(ro.Just(1, 2, 3)).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectComplete().
		Verify()

	// complete on error
	Assert[int](t).
		Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
			destination.Next(1)
			destination.Next(2)
			destination.Next(3)
			destination.Error(assert.AnError)
			destination.Next(4)
			return nil
		})).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectError(assert.AnError).
		Verify()

	// async observable
	Assert[int](t).
		Source(ro.NewObservable(func(destination ro.Observer[int]) ro.Teardown {
			go func() {
				destination.Next(1)
				destination.Next(2)
				destination.Next(3)
				destination.Error(assert.AnError)
				destination.Next(4)
			}()
			return nil
		})).
		ExpectNext(1).
		ExpectNext(2).
		ExpectNext(3).
		ExpectError(assert.AnError).
		Verify()
}
