---
title: 🚀 Getting started
description: Let's discover samber/ro in less than 5 minutes.
sidebar_position: 1
---

# 🚀 Getting started

Welcome to `samber/ro`! This guide will help you get started with reactive programming in Go. You'll learn the core concepts and see practical examples.

:::tip Quick Start

This guide is designed to get you up and running in under 5 minutes. Each example builds on previous concepts.

:::

## Installation

Make sure you have Go 1.18+ installed. The library uses modern Go features like generics.

```bash
go get -u github.com/samber/ro
```

## Your First Observable Stream

:::tip Hello World

Let's start with a simple example that creates a stream of values and processes them. This demonstrates the core reactive pattern: create, transform, and subscribe.

:::

```go
package main

import (
    "fmt"
    "time"

    "github.com/samber/ro"
)

func main() {
    // Create a simple stream
    observable := ro.Pipe2(
        ro.Interval(1 * time.Second),
        ro.Take[int64](5),
        ro.Map(func(x int64) string {
            return fmt.Sprintf("Tick: %d", x)
        }),
    )

    // Subscribe and print values
    subscription := observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s)
    }))

    // Wait for completion
    subscription.Wait()
}
```

**Output:**
```
Tick: 0
Tick: 1
Tick: 2
Tick: 3
Tick: 4
```

## Core Concepts

These four concepts are the building blocks of reactive programming with `ro`:

### 1. Observables

:::tip Data Sources

An [`Observable`](./core/observable) is a stream of values that can be observed over time:

:::

```go
// Create from values
numbers := ro.Just(1, 2, 3, 4, 5)

// Create from a slice
letters := ro.FromSlice([]string{"a", "b", "c"})
```

### 2. Operators

[Operators](./core/operators) transform, filter, or combine streams:

```go
// Chain operators with ro.Pipe
result := ro.Pipe2(
    ro.Range(0, 10),
    ro.Filter(func(x int64) bool {
        return x%2 == 0  // Keep only even numbers
    }),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("even-%d", x)  // Transform to string
    }),
)

result.Subscribe(ro.OnNext(func(s string) {
    fmt.Println(s) 
}))
// "even-0", "even-2", "even-4", "even-6", "even-8"
```

### 3. Subscriptions

:::warning Resource Management

[Subscriptions](./core/subscription) receive values from observables and manage cleanup:

:::

```go
subscription := observable.Subscribe(ro.NewObserver(
    func(value int) {           // OnNext
        fmt.Println("Received:", value)
    },
    func(err error) {           // OnError
        fmt.Println("Error:", err)
    },
    func() {                    // OnCompleted
        fmt.Println("Done!")
    },
))

// Cancel subscription if needed
subscription.Unsubscribe()
```

### 4. Multiple Subscriptions

Each call to `.Subscribe()` creates a new independent subscription that restarts the stream from the beginning. This is called a "cold" observable.

```go
source := ro.Just(1, 2, 3)

// First subscription
sub1 := source.Subscribe(ro.OnNext(func(x int) {
    fmt.Println("Subscriber 1:", x)
}))

// Second subscription - restarts from beginning
sub2 := source.Subscribe(ro.OnNext(func(x int) {
    fmt.Println("Subscriber 2:", x)
}))

// Output:
// Subscriber 1: 1
// Subscriber 1: 2
// Subscriber 1: 3
// Subscriber 2: 1
// Subscriber 2: 2
// Subscriber 2: 3
```

To share a single stream execution across multiple subscribers, use `.Share()` to create a hot observable (covered later in this guide).

## Common Operations

:::tip Daily Operations

These are the most frequently used operations in reactive programming:

:::

### Filtering Values

Filter operators let you select which values to process:

```go
obs := ro.Pipe1(
    ro.Range(0, 10),
    ro.Filter(func(x int64) bool {
        return x > 5  // Keep values greater than 5
    }),
)
// Output: 6, 7, 8, 9
```

### Transforming Data

:::tip Data Mapping

Transform operators convert values from one type to another:

:::

```go
obs := ro.Pipe1(
    ro.Just("apple", "banana", "cherry"),
    ro.Map(func(s string) string {
        return strings.ToUpper(s)
    }),
)
// Output: APPLE, BANANA, CHERRY
```

### Combining Streams

Combine multiple streams into one:

```go
stream1 := ro.Just(1, 2, 3)
stream2 := ro.Just(4, 5, 6)

obs := ro.Concat(stream1, stream2)
// Output: 1, 2, 3, 4, 5, 6
```

### Error Handling

:::warning Robust Applications

Handle errors gracefully to prevent application crashes:

:::

```go
riskyStream := ro.Pipe2(
    ro.Just(1, 2, 3, 4, 5),
    ro.MapErr(func(x int) (int, error) {
        if x == 3 {
            return 0, fmt.Errorf("error at %d", x)
        }
        return x * 2, nil
    }),
    ro.Catch(func(err error) ro.Observable[int] {
        fmt.Println("Recovered from error:", err)
        return ro.Just(42)  // Fallback value
    }),
)
```

## Real-world Example: API Rate Limiting

:::tip Practical Application

This example shows how to handle API calls with rate limiting and retry logic - a common real-world scenario.

:::

```go
package main

import (
    "fmt"
    "net/http"
    "time"

    "github.com/samber/ro"
)

func fetchUser(id int) (string, error) {
    // Simulate API call
    time.Sleep(100 * time.Millisecond)
    return fmt.Sprintf("user-%d", id), nil
}

func main() {
    // Create a stream of user IDs
    userIds := ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

    // Process with rate limiting
    userStream := ro.Pipe3(
        userIds,
        ro.Map(fetchUser),
        ro.DelayEach[string](200 * time.Millisecond),  // 200ms pause between items
        ro.RetryWithConfig(RetryConfig{MaxRetries: 2}),  // Retry failed requests
    )

    // Subscribe and collect results
    var results []string
    subscription := userStream.Subscribe(ro.NewObserver(
        func(user string) {
            results = append(results, user)
            fmt.Println("Fetched:", user)
        },
        func(err error) {
            fmt.Println("Error:", err)
        },
        func() {
            fmt.Println("All users fetched!")
            fmt.Println("Results:", results)
        },
    ))

    // Wait for completion
    subscription.Wait()
}
```

## Creating Custom Operators

Create reusable operators to encapsulate common transformations:

```go
// Custom operator that squares numbers
func Square[T constraints.Integer](observable ro.Observable[T]) ro.Observable[T] {
    return ro.Map(func(x T) T {
        return x * x
    })(observable)
}

func main() {
    result := Square(ro.Just(1, 2, 3, 4, 5))

    result.Subscribe(ro.OnNext(func(x int) {
        fmt.Println(x)  // 1, 4, 9, 16, 25
    }))
}
```

## Hot vs Cold Observables

:::warning Stream Behavior

Understanding the difference between hot and cold observables is crucial for building correct reactive applications:

:::

### Cold Observables (default)

Each subscriber gets their own independent stream:

```go
cold := ro.Just(1, 2, 3)

// Each subscriber sees the same values independently. Consumption starts on subscription.
cold.Subscribe(ro.OnNext(func(x int) { fmt.Println("Sub1:", x) }))
cold.Subscribe(ro.OnNext(func(x int) { fmt.Println("Sub2:", x) }))
```

### Hot Observables

:::tip Shared Execution

Multiple subscribers share the same stream. See [Subject](./core/subject) for more details.

:::

```go
// Create a hot observable from a cold one
hot := ro.Connectable(ro.Just(1, 2, 3))

// Both subscribers share the same sequence simultaneously
sub1 := hot.Subscribe(ro.OnNext(func(x int) { fmt.Println("Sub1:", x) }))
sub2 := hot.Subscribe(ro.OnNext(func(x int) { fmt.Println("Sub2:", x) }))

// Start subscription
subscription := connectable.Connect()
```

## Best Practices

:::tip Production Ready

Follow these practices to write maintainable and robust reactive code:

:::

### 1. Use Pipeline Operators

Pipeline operators promote clean, reusable code:
```go
// Good: Composable pipeline
pipeline := ro.Pipe3(
    source,
    ro.Filter(predicate),
    ro.Map(transformer),
    ro.Retry(3),
)

// Reusable pipeline
result1 := pipeline(stream1)
result2 := pipeline(stream2)
```

### 2. Handle Errors Gracefully

:::warning Error Recovery

Always handle errors to prevent application crashes:

:::
```go
stream := ro.Pipe2(
    riskyOperation,
    ro.Catch(func(err error) ro.Observable[string] {
        // Log error and provide fallback
        log.Println("Operation failed:", err)
        return fallbackStream
    }),
)
```

### 3. Manage Resources

:::danger Resource Leaks

Clean up resources to prevent memory leaks:

:::
```go
// Clean up resources when done
subscription := stream.Subscribe(observer)
defer subscription.Unsubscribe()
```

### 4. Avoid Memory Leaks

:::danger Bounded Streams

Always bound infinite streams to prevent memory exhaustion:

:::
```go
// Use Take to limit infinite streams
obs1 := ro.Pipe1(
    source,
    ro.Take[int](10),  // Only 10 values
)

// Use TakeUntil with timeout
obs2 := ro.Pipe1(
    source,
    stream.TakeUntil[int](ro.Timer(30*time.Second))
)
```

## Next Steps

:::tip Continue Learning

Now that you understand the basics, explore:

- **[Operators Reference](./operator/creation.md)**: Learn about all available operators
- **[Examples]**: Check out practical examples in the examples directory
- **[Comparison Guides](./comparison/lo-vs-ro)**: See how `samber/ro` compares to `channel`, `iter`, and `samber/lo`
- **[Advanced Patterns](./core/subject)**: `Subjects`, backpressure, and custom operators

:::

Happy streaming! 🚀
