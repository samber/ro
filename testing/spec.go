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

package rotesting

import (
	"context"

	"github.com/samber/ro"
)

// AssertSpec is an interface that defines the methods to assert the behavior of an
// observable sequence. It is inspired by Flux.
//
// Implementing this interface is optional. It is used to provide a more fluent API
// across different testing frameworks.
type AssertSpec[T any] interface {
	Source(source ro.Observable[T]) AssertSpec[T]
	ExpectNext(value T, msgAndArgs ...any) AssertSpec[T]
	ExpectNextSeq(items ...T) AssertSpec[T]
	ExpectError(err error, msgAndArgs ...any) AssertSpec[T]
	ExpectComplete(msgAndArgs ...any) AssertSpec[T]
	Verify()
	VerifyWithContext(ctx context.Context)
}
