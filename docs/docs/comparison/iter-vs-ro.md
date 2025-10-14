---
title: iter vs ro
description: Compare Go's iter package vs samber/ro reactive streams
sidebar_position: 4
---

# 🔁 `iter` vs `samber/ro`

Go's `iter` package (introduced in Go 1.23) and `samber/ro` both provide ways to work with sequences of values, but they serve different purposes and follow different paradigms:

- **Go iter package**: **Pull-based** - the consumer controls when to pull values
- **samber/ro**: **Push-based** - the producer pushes values to consumers

This comparison explores the fundamental differences between Go's standard iteration and reactive streams.

## Key Differences

:::tip Core Distinctions

### Concurrency Model
- **iter**: Synchronous by design
- **ro**: Asynchronous and concurrent by nature

### Data Flow
- **iter**: Sequential, one-time iteration
- **ro**: Continuous streams with multiple subscribers

### Error Handling
- **iter**: None
- **ro**: Error propagation, retry

:::

The fundamental difference is in the data flow model - pull vs push - which affects everything from concurrency to error handling.

## Code Comparison

### Basic Iteration

**Go iter**:

:::

The consumer controls when values are pulled using the standard `range` keyword.
```go
package main

import (
    "fmt"
    "iter"
)

func main() {
    // Define a sequence using iter.Seq
    numbers := func(yield func(int) bool) {
        for i := 1; i <= 5; i++ {
            if !yield(i) {
                break
            }
        }
    }

    // Pull values using range
    for n := range numbers {
        fmt.Println(n) // 1, 2, 3, 4, 5
    }
}
```

:::warning Push-based Streams

**samber/ro**:
```go
package main

import (
    "fmt"
    "github.com/samber/ro"
)

func main() {
    // Create an observable stream
    observable := ro.Just(1, 2, 3, 4, 5)

    // Subscribe to receive pushed values
    observable.Subscribe(ro.OnNext(func(n int) {
        fmt.Println(n) // 1, 2, 3, 4, 5
    }))
}
```

:::

Values are pushed to subscribers automatically, creating a reactive flow.

### Transformations

:::tip Manual Transformations

**Go iter**:

:::

With `iter`, you must implement transformation functions manually.
```go
// Map function for iter
func Map[V, W any](seq iter.Seq[V], f func(V) W) iter.Seq[W] {
    return func(yield func(W) bool) {
        for v := range seq {
            if !yield(f(v)) {
                break
            }
        }
    }
}

// Filter function for iter
func Filter[V any](seq iter.Seq[V], f func(V) bool) iter.Seq[V] {
    return func(yield func(V) bool) {
        for v := range seq {
            if f(v) && !yield(v) {
                break
            }
        }
    }
}

// Usage
numbers := func(yield func(int) bool) {
    for i := 0; i < 10; i++ {
        if !yield(i) {
            return
        }
    }
}

evens := Map(Filter(numbers, func(n int) bool {
    return n%2 == 0
}), func(n int) string {
    return fmt.Sprintf("even-%d", n)
})

for result := range evens {
    fmt.Println(result)
}
```

**samber/ro**:
```go
// Built-in operators
observable := ro.Pipe2(
    ro.Range(0, 10),
    ro.Filter(func(n int) bool {
        return n%2 == 0
    }),
    ro.Map(func(n int) string {
        return fmt.Sprintf("even-%d", n)
    }),
)

observable.Subscribe(ro.OnNext(func(result string) {
    fmt.Println(result)
}))
```

:::

`samber/ro` provides a rich set of built-in operators for common transformations.

### Async Operations

:::danger Synchronous Limitation

**Go iter** (synchronous only):

:::

The `iter` package is designed for synchronous operations only.
```go
// iter doesn't support async operations natively
func processData(data []int) iter.Seq[string] {
    return func(yield func(string) bool) {
        for _, item := range data {
            // This blocks the entire iteration
            result := expensiveSyncOperation(item)
            if !yield(result) {
                break
            }
        }
    }
}

// Blocking iteration
for result := range processData([]int{1, 2, 3}) {
    fmt.Println(result)
}
```

:::tip Async Native

**samber/ro** (asynchronous by default):

:::

Reactive programming is inherently asynchronous, perfect for real-time applications.
```go
var pipeline = ro.PipeOp2(
    ro.Map(expensiveAsyncOperation),
    ro.RetryWithConfig(RetryConfig{MaxRetries: 3}),
)

func main() {
    observable := pipeline(ro.Just(1, 2, 3))

    // Non-blocking subscription
    _ = observable.Subscribe(ro.OnNext(func(result string) {
        fmt.Println(result)
    }))
}
```

### Multiple Consumers

:::warning Single Consumer

**Go iter** (single consumer):

:::

Each iteration consumes the sequence, making it difficult to share data streams.
```go
func generateNumbers() iter.Seq[int] {
    return func(yield func(int) bool) {
        for i := 1; i <= 5; i++ {
            if !yield(i) {
                return
            }
        }
    }
}

// Each iteration consumes the sequence
seq := generateNumbers()

// First consumer
for n := range seq {
    fmt.Println("Consumer 1:", n)
}

// Second consumer
for n := range seq {
    fmt.Println("Consumer 2:", n)
}

// Both subscribers receive: 1, 2, 3, 4, 5
```

**samber/ro** (multiple subscribers):

:::

Multiple subscribers can receive the same data stream simultaneously.
```go
// Hot observable - multiple subscribers get all values
observable := ro.Just(1, 2, 3, 4, 5)

// Multiple subscribers
observable.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Subscriber 1:", n)
}))

observable.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Subscriber 2:", n)
}))

// Both subscribers receive: 1, 2, 3, 4, 5
```

## Advanced Features

### Error Handling

:::danger No Error Handling

**Go iter**:

:::

Error handling is not built into the `iter` paradigm and requires manual intervention.
```go
func riskyOperation() iter.Seq[int] {
    return func(yield func(int) bool) {
        for i := 1; i <= 5; i++ {
            if i == 3 {
                // Can't easily propagate errors through yield
                panic("error")
            }
            if !yield(i) {
                return
            }
        }
    }
}
```

:::tip Built-in Error Handling

**samber/ro**:
```go
func createRiskyStream() ro.Observable[int] {
    return ro.Pipe2(
        ro.Range(1, 6),
        ro.MapErr(func(i int) (int, error) {
            if i == 3 {
                return 0, fmt.Errorf("error at %d", i)
            }
            return i, nil
        }),
    )
}

// Built-in error handling
createRiskyStream().Subscribe(ro.Observer[int]{
    OnNext: func(n int) {
        fmt.Println("Received:", n)
    },
    OnError: func(err error) {
        fmt.Println("Error:", err) // Handles error at 3
    },
})
```

:::

Reactive streams have first-class support for error propagation and recovery.

### Time-based Operations

:::warning Manual Implementation

**Go iter** (no built-in time operations):

:::

Time-based operations require manual implementation with channels and goroutines.
```go
// Manual time-based operations are complex and non-idiomatic
func timedSequence() iter.Seq[int] {
    return func(yield func(int) bool) {
        ticker := time.NewTicker(time.Second)
        defer ticker.Stop()

        counter := 0
        for {
            select {
            case <-ticker.C:
                counter++
                if !yield(counter) {
                    return
                }
            }
        }
    }
}
```

**samber/ro** (native time operators):

:::

Built-in time operators make it easy to work with temporal data streams.
```go
// Built-in time operations
observable := ro.Pipe3(
    ro.Interval(time.Second),
    ro.Take(42),
    ro.Map(func(tick int) string {
        return fmt.Sprintf("tick-%d", tick)
    }),
)

observable.Subscribe(ro.OnNext(func(msg string) {
    fmt.Println(msg) // "tick-1", "tick-2", etc. every second
}))
```

## When to Use Which

:::info Decision Guide

### Use Go iter when:
- Working with synchronous sequences
- Need standard library compatibility
- Writing simple iteration logic
- Memory efficiency is critical
- No need for complex async operations

### Use samber/ro when:
- Handling real-time events
- Need async processing
- Building reactive applications
- Multiple subscribers required
- Complex error handling needed

:::

Consider your specific requirements for synchronicity, error handling, and concurrency when choosing between these approaches.

## Performance Characteristics

:::warning Performance Considerations

| Aspect           | Go iter              | samber/ro               |
| ---------------- | -------------------- | ----------------------- |
| **Memory Usage** | Low (lazy producing) | Low (lazy producing)    |
| **Latency**      | Zero                 | medium (small overhead) |
| **CPU Usage**    | Predictable          | Predictable             |
| **Concurrency**  | None                 | Built-in                |
| **Backpressure** | Manual               | Automatic               |

:::

Both approaches offer lazy evaluation, but `ro` provides built-in concurrency and backpressure management.

## Feature Comparison

:::info Feature Matrix

| Feature              | Go iter | samber/ro |
| -------------------- | ------- | --------- |
| Pull-based Iteration | ✅       | ❌         |
| Push-based Streams   | ❌       | ✅         |
| Async Processing     | ❌       | ✅         |
| Error Handling       | Manual  | Built-in  |
| Time Operations      | ❌       | ✅         |
| Multiple Subscribers | ❌       | ✅         |
| Backpressure         | ❌       | ✅         |
| Standard Library     | ✅       | ❌         |
| Zero Dependencies    | ✅       | ❌         |

:::

Choose `iter` for standard library integration and simple iteration, or `ro` for reactive programming capabilities.

## Migration Examples

:::tip Migration Guide

### From iter to ro

**Before (iter)**:

:::

Converting from `iter` to `ro` typically involves replacing pull-based iteration with push-based streams.
```go
func processItems(items []string) iter.Seq[string] {
    return func(yield func(string) bool) {
        for _, item := range items {
            processed := strings.ToUpper(item)
            if !yield(processed) {
                return
            }
        }
    }
}

for result := range processItems([]string{"a", "b", "c"}) {
    fmt.Println(result)
}
```

:::info Reactive Approach

**After (ro)**:

:::

Notice how the reactive approach simplifies the code and provides more flexibility.
```go
func processItems(items []string) ro.Observable[string] {
    return ro.Pipe2(
        ro.Just(items...),
        ro.Map(func(item string) string {
            return strings.ToUpper(item)
        }),
    )
}

processItems([]string{"a", "b", "c"}).Subscribe(ro.OnNext(func(result string) {
    fmt.Println(result)
}))
```

Go's `iter` package is excellent for synchronous iteration and sequences, while `samber/ro` provides powerful reactive capabilities for asynchronous, event-driven programming. Choose based on your specific use case and requirements.

:::tip Learn More

- Explore [Observable basics](../core/observable) for reactive concepts
- Learn about [backpressure](../glossary#Backpressure) in reactive systems
- Compare with [Go channels](../comparison/channels-vs-ro) for another concurrency approach

:::
