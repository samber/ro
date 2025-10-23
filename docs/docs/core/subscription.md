---
title: Subscription
description: Learn about Subscription - resource management and cleanup for reactive streams in samber/ro
sidebar_position: 4
---

# 🔄 Subscription

A **Subscription** represents the ongoing execution of an [`Observable`](./observable) and provides the interface for managing resources, cleanup, and cancellation. Subscriptions are essential for proper resource management in reactive programming.

## What is a Subscription?

A `Subscription` is:
- **A resource manager**: Handles cleanup of `Observable` executions
- **A cancellation token**: Allows stopping ongoing operations
- **A lifecycle controller**: Manages the execution state of Observables
- **Thread-safe**: Safe for concurrent use across multiple goroutines

## Subscription Interface

The `Subscription` interface provides methods for managing `Observable` execution:

```go
type Subscription interface {
    Unsubscribable

    Add(teardown Teardown)
    AddUnsubscribable(unsubscribable Unsubscribable)
    IsClosed() bool
    Wait() // Note: using .Wait() is not recommended.
}

type Unsubscribable interface {
    Unsubscribe()
}

type Teardown func()
```

## Creating Subscriptions

### Basic Subscription

:::warning Resource Management

Always capture the returned subscription to manage cleanup. This is essential for preventing resource leaks in reactive applications.

:::

```go
// Create an Observable
observable := ro.Interval(1 * time.Second)

// Subscribe and get the subscription
subscription := observable.Subscribe(ro.OnNext(func(tick int64) {
    fmt.Println("Tick:", tick)
}))

// Cancel the subscription
subscription.Unsubscribe()
```

### Subscription with Teardown

:::tip Cleanup Order

Use teardown functions to clean up resources when subscriptions are cancelled. Functions execute in reverse order (LIFO) - last added, first executed.

:::

```go
// Create a subscription with cleanup logic
subscription := ro.NewSubscription(func() {
    fmt.Println("Cleaning up resources...")
    // Close files, database connections, etc.
})

// Add additional cleanup functions
subscription.Add(func() {
    fmt.Println("Additional cleanup...")
})

// Unsubscribe to trigger all cleanup functions
subscription.Unsubscribe()
// Output:
// Additional cleanup...
// Cleaning up resources...
```

## Resource Management

### Adding Teardown Functions

:::danger Real Resource Management

This example demonstrates proper file handling with resource cleanup. Always ensure files, network connections, and other resources are properly closed when subscriptions are cancelled.

:::

```go
// Observable that reads from a file and requires cleanup
func createFileObservable(filePath string) ro.Observable[string] {
    return ro.NewObservable(func(observer ro.Observer[string]) ro.Teardown {
        // Open the file
        file, err := os.Open(filePath)
        if err != nil {
            observer.Error(err)
            return nil
        }

        scanner := bufio.NewScanner(file)
        done := make(chan struct{})

        go func() {
            for scanner.Scan() {
                select {
                case <-done:
                    return
                default:
                    observer.Next(scanner.Text())
                }
            }

            if err := scanner.Err(); err != nil {
                observer.Error(err)
            } else {
                observer.Complete()
            }
        }()

        // Return cleanup function
        return func() {
            close(done)
            file.Close()
            fmt.Printf("Closed file: %s\n", filePath)
        }
    })
}

// Subscribe and manage file resources
func main() {
    // Create observable from file
    fileObservable := createFileObservable(tmpFile)
    subscription := fileObservable.Subscribe(ro.OnNext(func(line string) {
        fmt.Printf("Line: %s\n", line)
    }))

    // Wait for completion, then cleanup is automatic
    subscription.Wait()
}
// Output:
// Line: Hello
// Line: World
// Line: Reactive
// Line: Programming
// Closed file: /tmp/ro-example-1234567890.txt
```

### Combining Multiple Subscriptions

:::info Coordinated Cleanup

Group related subscriptions together for coordinated cleanup. This pattern is useful when managing multiple streams that should be cancelled together, such as in a user interface with multiple data streams.

:::

```go
// Create multiple observables
obs1 := ro.Pipe1(ro.Interval(1 * time.Second), ro.Take[int64](3))
obs2 := ro.Pipe1(ro.Interval(2 * time.Second), ro.Take[int64](2))

// Subscribe to observables and add to main subscription
sub1 := obs1.Subscribe(ro.OnNext(func(n int64) {
    fmt.Println("Stream 1:", n)
}))
sub2 := obs2.Subscribe(ro.OnNext(func(n int64) {
    fmt.Println("Stream 2:", n)
}))

// Create main subscription
mainSubscription := ro.NewSubscription(nil)
mainSubscription.AddUnsubscribable(sub1)
mainSubscription.AddUnsubscribable(sub2)

// Cancel all subscriptions at once
time.AfterFunc(5 * time.Second, func() {
    fmt.Println("Canceling all subscriptions")
    mainSubscription.Unsubscribe()
})
```

## Subscription Lifecycle

```go
subscription := ro.Interval(1 * time.Second).Subscribe(ro.OnNext(func(tick int64) {
    fmt.Printf("Tick: %d (closed: %t)\n", tick, subscription.IsClosed())
}))

// Check state
fmt.Println("Subscription closed:", subscription.IsClosed()) // false

// Cancel and check again
subscription.Unsubscribe()
fmt.Println("Subscription closed:", subscription.IsClosed()) // true
```

## Error Handling in Subscriptions

:::warning Panic Recovery

Subscriptions automatically handle panics in teardown functions, preventing application crashes during cleanup. This provides robust error handling for resource management.

:::

```go
// Subscriptions automatically handle panics in teardown functions
subscription := ro.NewSubscription(func() {
    panic("something went wrong in cleanup!")
})

subscription.Add(func() {
    fmt.Println("This will still execute")
})

subscription.Unsubscribe()
```

## Best Practices

### 1. Never ignore the returned Subscriptions

:::danger Resource Leaks

Ignoring returned subscriptions can lead to memory leaks and unmanaged resources. Always capture and properly manage subscriptions.

:::

```go
// Good: Explicit subscription management
func processData() {
    subscription := observable.Subscribe(observer)
    defer subscription.Unsubscribe()

    // Do other work
}

// Risky: Potential resource leak
func processDataRisky() {
    observable.Subscribe(observer) // No cleanup
}
```

### 2. Use Context for Cancellation

:::tip Context-Aware Cleanup

Use Go's context package for coordinated cancellation across multiple operations. This pattern integrates well with standard Go practices for cancellation and timeouts.

:::

```go
// Good: Context-aware cancellation
func processWithContext(ctx context.Context) {
    subscription := observable.Subscribe(ro.NewObserver(
        func(value int) {
            select {
            case <-ctx.Done():
                subscription.Unsubscribe()
                return
            default:
                fmt.Println(value)
            }
        },
        // ...
    ))

    <-ctx.Done()
    subscription.Unsubscribe()
}
```

### 3. Avoid Blocking Operations

:::warning Anti-Pattern

Using `.Wait()` goes against reactive programming principles. Instead, use non-blocking subscriptions and handle results asynchronously through [`Observer`](./observer) callbacks.

:::

```go
// Questionable: Blocking wait
func processBlocking() {
    subscription := observable.Subscribe(observer)
    subscription.Wait() // Against reactive principles
}
```

### 4. Group Related Subscriptions

:::info Composite Pattern

Group related subscriptions to manage complex systems with multiple data streams. This pattern is particularly useful in microservices, real-time applications, and user interfaces.

:::

```go
// Good: Group related operations
func processMultipleStreams() {
    main := ro.NewSubscription(nil)

    stream1 := ro.Interval(1 * time.Second)
    stream2 := ro.Interval(2 * time.Second)

    sub1 := stream1.Subscribe(observer1)
    sub2 := stream2.Subscribe(observer2)

    main.AddUnsubscribable(sub1)
    main.AddUnsubscribable(sub2)

    // Single cancellation point
    defer main.Unsubscribe()
}
```

### 5. Use defer

:::tip Automatic Cleanup

Ensure the cleanup operation will be done to prevent any stream leakage. Using `defer` guarantees cleanup even if panics occur, making your reactive code more robust.

:::

Ensure the cleanup operation will be done to prevent any stream leakage.

```go
obs := ro.Interval(1 * time.Second)

sub := obs.Subscribe(...)

// .Subscribe(...) should always be followed by a defered
// unsubscription. Even if .Subscribe(...) is expected to
// block until stream completion.
defer sub.Unsubscribe()
```

Subscriptions provide a powerful, centralized way to manage resources and control the lifecycle of Observable executions in reactive programming, making it easier to write robust, resource-safe applications.
