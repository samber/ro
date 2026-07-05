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


package rolog

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log"
	"os"

	"github.com/samber/ro"
)

func ExampleLog() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// Log all notifications (Next, Error, Complete)
	observable := ro.Pipe1(
		ro.Just(1, 2, 3, 4, 5),
		Log[int](),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// ro.Next: 1
	// ro.Next: 2
	// ro.Next: 3
	// ro.Next: 4
	// ro.Next: 5
	// ro.Complete
}

func ExampleLogWithPrefix() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// Log with a custom prefix
	observable := ro.Pipe1(
		ro.Just("hello", "world", "golang"),
		LogWithPrefix[string]("[MyApp]"),
	)

	subscription := observable.Subscribe(ro.NoopObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// [MyApp] ro.Next: hello
	// [MyApp] ro.Next: world
	// [MyApp] ro.Next: golang
	// [MyApp] ro.Complete
}

func ExampleLog_withError() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// Log including error notifications
	observable := ro.Pipe1(
		ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
			observer.Next(1)
			observer.Next(2)
			observer.Error(errors.New("something went wrong"))
			observer.Next(3) // This won't be emitted due to error
			return nil
		}),
		Log[int](),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// ro.Next: 1
	// ro.Next: 2
	// ro.Error: something went wrong
}

// func ExampleFatalOnError() {
// 	br := bufio.NewWriter(os.Stdout)
// 	log.SetOutput(br)
// 	log.SetFlags(0)
// 	defer br.Flush()

// 	// Fatal on error (this would terminate the program in real usage)
// 	observable := ro.Pipe1(
// 		ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
// 			observer.Next(1)
// 			observer.Next(2)
// 			observer.Error(errors.New("critical error"))
// 			return nil
// 		}),
// 		FatalOnError[int](),
// 	)

// 	subscription := observable.Subscribe(ro.NoopObserver[int]())
// 	defer subscription.Unsubscribe()
// }

// func ExampleFatalOnErrorWithPrefix() {
// 	br := bufio.NewWriter(os.Stdout)
// 	log.SetOutput(br)
// 	log.SetFlags(0)
// 	defer br.Flush()

// 	// Fatal on error with custom prefix
// 	observable := ro.Pipe1(
// 		ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
// 			observer.Next(1)
// 			observer.Error(errors.New("database connection failed"))
// 			return nil
// 		}),
// 		FatalOnErrorWithPrefix[int]("[Database]"),
// 	)

// 	subscription := observable.Subscribe(ro.NoopObserver[int]())
// 	defer subscription.Unsubscribe()
// }

func ExampleLog_inPipeline() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// Use logging in a complex pipeline
	observable := ro.Pipe3(
		ro.Just(1, 2, 3, 4, 5),
		ro.Filter(func(n int) bool { return n%2 == 0 }), // Keep even numbers
		LogWithPrefix[int]("[Filter]"),
		ro.Map(func(n int) string { return fmt.Sprintf("Even: %d", n) }),
	)

	subscription := observable.Subscribe(ro.NoopObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// [Filter] ro.Next: 2
	// [Filter] ro.Next: 4
	// [Filter] ro.Complete
}

func ExampleLog_withContext() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// Log with context-aware operations
	ctx := context.Background()

	observable := ro.Pipe1(
		ro.Just("context", "aware", "logging"),
		LogWithPrefix[string]("[Context]"),
	)

	subscription := observable.SubscribeWithContext(ctx, ro.NoopObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// [Context] ro.Next: context
	// [Context] ro.Next: aware
	// [Context] ro.Next: logging
	// [Context] ro.Complete
}

func ExampleFatalOnError() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// FatalOnError passes items through when no error occurs.
	// When an error is received, it calls log.Fatal (which exits the program).
	observable := ro.Pipe1(
		ro.Just(1, 2, 3),
		FatalOnError[int](),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
}

func ExampleFatalOnErrorWithPrefix() {
	br := bufio.NewWriter(os.Stdout)
	log.SetOutput(br)
	log.SetFlags(0)
	defer br.Flush()

	// FatalOnErrorWithPrefix passes items through when no error occurs.
	// When an error is received, it calls log.Fatal with the given prefix.
	observable := ro.Pipe1(
		ro.Just("a", "b", "c"),
		FatalOnErrorWithPrefix[string]("[MyApp]"),
	)

	subscription := observable.Subscribe(ro.NoopObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
}
