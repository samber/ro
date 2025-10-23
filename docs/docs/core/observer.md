---
title: Observer
description: Learn about Observer - the consumer interface for reactive streams in samber/ro
sidebar_position: 2
---

# 👁️ Observer

An **Observer** is the consumer side of reactive programming in `samber/ro`. It receives notifications from [`Observable`](./observable) through three essential methods: `Next`, `Error`, and `Complete`. `Observer` are the destination for values emitted by `Observable`.

## What is an Observer?

An `Observer` is:
- **A consumer of values**: It receives values emitted by Observables
- **Notification handler**: It processes `Next`, `Error`, and `Complete` notifications
- **Stateful**: It tracks whether it's active, completed, or errored
- **Thread-safe**: Multiple goroutines can safely call Observer methods

## Observer Interface

The Observer interface defines three core methods `Next`, `Error`, and `Complete`, with `XxxxWithContext` variants.

```go
type Observer[T any] interface {
    // Next receives the next value from the Observable
    Next(value T)
    NextWithContext(ctx context.Context, value T)

    // Error receives an error notification (terminal)
    Error(err error)
    ErrorWithContext(ctx context.Context, err error)

    // Complete receives a completion notification (terminal)
    Complete()
    CompleteWithContext(ctx context.Context)

    // State checking methods
    IsClosed() bool
    HasThrown() bool
    IsCompleted() bool
}
```

## Creating Observers

### Complete Observer

Always use complete observers in production code to handle errors and completion signals properly.

```go
// Create a full Observer with all callbacks
observer := ro.NewObserver(
    func(value int) {
        fmt.Println("Received:", value)
    },
    func(err error) {
        fmt.Println("Error:", err)
    },
    func() {
        fmt.Println("Completed")
    },
)

// Use with an Observable
observable := ro.Just(1, 2, 3)
observable.Subscribe(observer)
// Output:
// Received: 1
// Received: 2
// Received: 3
// Completed
```

### Context-aware Observer

:::warning Context-aware Observer

Use context-aware observers when you need timeout control or cancellation. The context is passed through the entire pipeline and can be used to stop processing.

:::

```go
// Create an Observer with context support
observer := ro.NewObserverWithContext(
    func(ctx context.Context, value int) {
        fmt.Printf("Received %d with context\n", value)
    },
    func(ctx context.Context, err error) {
        fmt.Printf("Error %v with context\n", err)
    },
    func(ctx context.Context) {
        fmt.Println("Completed with context")
    },
)

observable.SubscribeWithContext(context.Background(), observer)
```

## Partial Observers

### Next-only Observer

Use `ro.OnNext()` when you only need to handle values and can ignore errors and completion signals for simple use cases.

```go
// Handle only Next values, ignore errors and completion
observer := ro.OnNext(func(value string) {
    fmt.Println("Got:", strings.ToUpper(value))
})

ro.Just("hello", "world").Subscribe(observer)
// Output: GOT: HELLO, GOT: WORLD
```

### Error-only Observer

:::warning Error Handling

Error-only observers are useful for logging or monitoring failure scenarios without processing the actual values. Always handle errors in production code.

:::

```go
// Handle only errors, ignore values
observer := ro.OnError(func(err error) {
    log.Printf("Operation failed: %v", err)
})

riskyObservable.Subscribe(observer)
```

### Complete-only Observer

Complete-only observers are useful for cleanup operations or triggering follow-up actions after a stream finishes.

```go
// Handle only completion
observer := ro.OnComplete(func() {
    fmt.Println("All processing completed")
})

longRunningObservable.Subscribe(observer)
```

## Observer Lifecycle

An Observer can be in one of three states:
1. **Active**: Ready to receive notifications (default state)
2. **Completed**: Received a Complete notification, no more notifications accepted
3. **Errored**: Received an Error notification, no more notifications accepted

```go
observer := ro.NewObserver(
    func(value int) {
        fmt.Println("State:", observer.IsClosed()) // false while active
        fmt.Println("Received:", value)
    },
    func(err error) {
        fmt.Println("State:", observer.IsClosed()) // true
        fmt.Println("HasThrown:", observer.HasThrown()) // true
        fmt.Println("Error:", err)
    },
    func() {
        fmt.Println("State:", observer.IsClosed()) // true
        fmt.Println("IsCompleted:", observer.IsCompleted()) // true
    },
)
```

## Error Handling in Observers

### Panic Recovery

:::danger Panic Recovery

Observers automatically recover from panics in callback functions, preventing application crashes.

:::

```go
observer := ro.NewObserver(
    func(value int) {
        if value == 3 {
            panic("something went wrong!")
        }
        fmt.Println("Value:", value)
    },
    func(err error) {
        fmt.Println("Recovered error:", err) // Handles the panic
    },
    func() {
        fmt.Println("Completed")
    },
)

ro.Just(1, 2, 3, 4).Subscribe(observer)
// Output:
// Value: 1
// Value: 2
// Recovered error: something went wrong!
```

### State After Error

Once an Observer receives an error, it rejects further notifications:

```go
observer := ro.NewObserver(
    func(value int) {
        fmt.Println("Got:", value)
    },
    func(err error) {
        fmt.Println("Error:", err)
    },
    func() {
        fmt.Println("Completed")
    },
)

// Send error
observer.Error(fmt.Errorf("network failure"))

// Try to send more values (will be ignored)
observer.Next(42)      // Ignored
observer.Complete()    // Ignored

fmt.Println("IsClosed:", observer.IsClosed()) // true
fmt.Println("HasThrown:", observer.HasThrown()) // true
```

## Utility Observers

### No-op Observer

```go
// Observer that does nothing
noop := ro.NoopObserver[string]()
ro.Just("hello", "world").Subscribe(noop) // Values are silently consumed
// Output:
```

### Print Observer (Debugging)

```go
// Observer that prints all notifications for debugging
debugObserver := ro.PrintObserver[string]()
ro.Just("hello", "world").Subscribe(debugObserver)
// Output:
// Next: hello
// Next: world
// Completed
```

## Observer Best Practices

### 1. Always Handle Errors

:::warning Best Practice

In production code, always handle all three observer callbacks (Next, Error, Complete) to ensure proper error handling and resource cleanup.

:::

```go
// Good: Handle errors
observer := ro.NewObserver(
    func(value int) { /* process value */ },
    func(err error) { /* handle error */ },
    func() { /* handle completion */ },
)

// Risky: No error handling in potentially failing operations
observer := ro.OnNext(func(value int) { /* process value */ })
```

:::info Operator-based Processing

While Observers can handle side effects, it's better to perform async operations using operators like `.FlatMap` to maintain the reactive pipeline benefits.

:::

### 2. Handle Async Operations with Operators

While Observers can handle side effects, it's better to perform async operations using operators like `.FlatMap`:

```go
// Good: Keep OnNext method focused on terminal consumption
observer := ro.OnNext(func(value int) {
    result := value * 2
    fmt.Println("Processed:", result)
})

// Avoid: Complex async operations in terminal Observer
observer := ro.OnNext(func(value int) {
    // This blocks the Observer and loses error handling benefits
    if err := writeToDatabase(value); err != nil {
        log.Println("Failed to save:", err) // No clean error propagation
    }
})

// Better: Use MapErr for async operations
// Chain operations in the pipeline
pipeline := ro.Pipe2(
    ro.Just(1, 2, 3),
    ro.Map(func(value int) int { return value * 2 }),
    ro.MapErr(func(value int) (int, error) {
        err := writeToDatabase(value)
        return value, err
    }),
)

pipeline.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Println("Saved to database:", value)
    },
    func(err error) {
        fmt.Println("Error saving:", err)
    },
    func() {
        fmt.Println("All saves completed")
    },
))
```

### 3. Handle Backpressure

:::tip Backpressure Advantage

In `samber/ro`, backpressure is handled naturally through blocking behavior. When you call `observer.Next()`, the call blocks until all downstream operators in the pipeline have processed the value. This ensures the stream is consumed sequentially and in the right order.

:::

```go
// Example: Processing values with backpressure
func createProcessingObservable() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 1; i <= 5; i++ {
            fmt.Printf("Emitting %d\n", i)

            // This blocks until downstream operators complete
            observer.Next(i)

            fmt.Printf("Downstream completed for %d\n", i)
        }
        observer.Complete()
        return nil
    })
}

// Create a pipeline with a slow operator
pipeline := ro.Pipe1(
    createProcessingObservable(),
    ro.Map(func(value int) int {
        // Simulate slow processing
        time.Sleep(100 * time.Millisecond)
        return value * 2
    }),
)

pipeline.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("Received: %d\n", value)
    },
    func(err error) {
        fmt.Println("Error:", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
// Output:
// Emitting 1
// Received: 2
// Downstream completed for 1
// Emitting 2
// Received: 4
// Downstream completed for 2
// ...
```

This blocking behavior ensures that:
- The producer waits for consumers to be ready
- Memory usage remains bounded
- No values are lost due to overflow
- The pipeline naturally regulates flow rate
- The stream is consumed in a sequential fashion
- The message order is preserved

Observers are the essential consumer interface in reactive programming, providing a clean, thread-safe way to handle streams of values with proper error handling and lifecycle management.
