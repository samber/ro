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


package rologrus

import (
	"bufio"
	"context"
	"errors"
	"os"

	"github.com/samber/ro"
	"github.com/sirupsen/logrus"
)

func ExampleLog() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.InfoLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	defer buff.Flush()

	observable := ro.Pipe1(
		ro.Just(1, 2, 3, 4, 5),
		Log[int](logger, logrus.InfoLevel),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// level=info msg="ro.Next: 1"
	// level=info msg="ro.Next: 2"
	// level=info msg="ro.Next: 3"
	// level=info msg="ro.Next: 4"
	// level=info msg="ro.Next: 5"
	// level=info msg=ro.Complete
}

func ExampleLogWithNotification() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.InfoLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	defer buff.Flush()

	observable := ro.Pipe1(
		ro.Just("Hello", "World", "Golang"),
		LogWithNotification[string](logger, logrus.InfoLevel),
	)

	subscription := observable.Subscribe(ro.NoopObserver[string]())
	defer subscription.Unsubscribe()

	// Output:
	// level=info msg=ro.Next value=Hello
	// level=info msg=ro.Next value=World
	// level=info msg=ro.Next value=Golang
	// level=info msg=ro.Complete
}

// func ExampleFatalOnError() {
// 	// Create a logger that writes to a buffer
// 	buff := bufio.NewWriter(os.Stdout)
// 	logger := logrus.New()
// 	logger.SetOutput(buff)
// 	logger.SetLevel(logrus.InfoLevel)
// 	logger.SetFormatter(&logrus.TextFormatter{
// 		DisableColors:    true,
// 		DisableTimestamp: true,
// 	})
// 	defer buff.Flush()

// 	// Create an observable that will emit an error
// 	observable := ro.Pipe1(
// 		ro.Throw[int](assert.AnError),
// 		FatalOnError[int](logger),
// 	)

// 	subscription := observable.Subscribe(ro.NoopObserver[int]())
// 	defer subscription.Unsubscribe()
// }

func ExampleLog_withContext() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.InfoLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	defer buff.Flush()

	ctx := context.WithValue(context.Background(), "request_id", "12345")

	observable := ro.Pipe1(
		ro.Just(1, 2, 3),
		Log[int](logger, logrus.InfoLevel),
	)

	subscription := observable.SubscribeWithContext(ctx, ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// level=info msg="ro.Next: 1"
	// level=info msg="ro.Next: 2"
	// level=info msg="ro.Next: 3"
	// level=info msg=ro.Complete
}

func ExampleLogWithNotification_withStructuredData() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.InfoLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	defer buff.Flush()

	type User struct {
		Name string
		Age  int
	}

	observable := ro.Pipe1(
		ro.Just(
			User{Name: "Alice", Age: 30},
			User{Name: "Bob", Age: 25},
			User{Name: "Charlie", Age: 35},
		),
		LogWithNotification[User](logger, logrus.InfoLevel),
	)

	subscription := observable.Subscribe(ro.NoopObserver[User]())
	defer subscription.Unsubscribe()

	// Output:
	// level=info msg=ro.Next value="{Alice 30}"
	// level=info msg=ro.Next value="{Bob 25}"
	// level=info msg=ro.Next value="{Charlie 35}"
	// level=info msg=ro.Complete
}

func ExampleLog_withDifferentLevels() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.DebugLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	defer buff.Flush()

	observable := ro.Pipe1(
		ro.Just(1, 2, 3),
		Log[int](logger, logrus.DebugLevel),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// level=debug msg="ro.Next: 1"
	// level=debug msg="ro.Next: 2"
	// level=debug msg="ro.Next: 3"
	// level=debug msg=ro.Complete
}

func ExampleFatalOnError() {
	// Create a logger that writes to a buffer
	buff := bufio.NewWriter(os.Stdout)
	logger := logrus.New()
	logger.SetOutput(buff)
	logger.SetLevel(logrus.FatalLevel)
	logger.SetFormatter(&logrus.TextFormatter{
		DisableColors:    true,
		DisableTimestamp: true,
	})
	// Prevent os.Exit(1) from terminating the process during the example.
	// logrus.Logger.ExitFunc defaults to os.Exit; override it to a no-op so
	// Fatal() only logs without stopping the test runner.
	logger.ExitFunc = func(int) {}
	defer buff.Flush()

	observable := ro.Pipe1(
		ro.Throw[int](errors.New("something went wrong")),
		FatalOnError[int](logger),
	)

	subscription := observable.Subscribe(ro.NoopObserver[int]())
	defer subscription.Unsubscribe()

	// Output:
	// level=fatal msg=ro.Error error="something went wrong"
}
