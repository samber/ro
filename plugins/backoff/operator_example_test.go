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

package robackoff

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/samber/ro"
)

func ExampleRetry_basic() {
	attempts := 0
	src := ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[int]) ro.Teardown {
		attempts++
		if attempts < 3 {
			dst.ErrorWithContext(ctx, errors.New("transient error"))
			return nil
		}
		dst.NextWithContext(ctx, 42)
		dst.CompleteWithContext(ctx)
		return nil
	})

	observable := ro.Pipe1(
		src,
		Retry[int](Constant(1*time.Millisecond)),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Values: %v\n", values)
	// Output: Values: [42]
}

func ExampleRetryWithConfig_exponential() {
	attempts := 0
	src := ro.NewUnsafeObservableWithContext(func(ctx context.Context, dst ro.Observer[string]) ro.Teardown {
		attempts++
		if attempts < 2 {
			dst.ErrorWithContext(ctx, errors.New("network timeout"))
			return nil
		}
		dst.NextWithContext(ctx, "success")
		dst.CompleteWithContext(ctx)
		return nil
	})

	observable := ro.Pipe1(
		src,
		RetryWithConfig[string](Config{
			MaxRetries: 3,
			Strategy:   Exponential(10*time.Millisecond, 2.0, 1*time.Second),
		}),
	)

	values, err := ro.Collect(observable)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Printf("Result: %v\n", values)
	// Output: Result: [success]
}

func ExampleRetryWithConfig_maxRetriesExceeded() {
	src := ro.Throw[int](errors.New("persistent failure"))

	observable := ro.Pipe1(
		src,
		RetryWithConfig[int](Config{
			MaxRetries: 2,
			Strategy:   Constant(1 * time.Millisecond),
		}),
	)

	_, err := ro.Collect(observable)
	fmt.Printf("Final error: %v\n", err)
	// Output: Final error: persistent failure
}

