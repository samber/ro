---
title: Observable
description: "Learn about Observable - the core concept of reactive programming in samber/ro"
sidebar_position: 1
---

# 📡 Observable

An **Observable** is the foundation of reactive programming. It represents a stream of values that can be observed over time, serving as both the data source and the factory for streams.

## What is an Observable?

An `Observable` is:
- **A producer of values**: It emits zero or more values over time
- **A stream factory**: Each subscription creates a new independent execution
- **Lazy by nature**: Values are produced only when subscribed to
- **Push-based**: Values are pushed to observers rather than pulled

An Observable can emit three types of notifications:
1. **Next**: A value from the sequence
2. **Error**: An error that terminates the stream
3. **Complete**: A signal that the stream has finished successfully

Once an Observable emits an Error or Complete notification, it will not emit any more values.

:::tip Hot vs Cold Observables

Understanding the difference between hot and cold observables is crucial:
- **Cold observables** (default): Each subscriber gets independent values
- **Hot observables**: Subscribers share the same execution

See [Subject](./subject) to learn how to create hot observables.
:::

## Creating Observables

### From Values

Create observables from fixed values. The `ro.Just()` operator emits each value immediately and sequentially to subscribers.

```go
// Create an Observable from a finite list of values
numbers := ro.Just(1, 2, 3, 4, 5)

// Subscribe to receive values
numbers.Subscribe(ro.OnNext(func(n int) {
    fmt.Println(n) // 1, 2, 3, 4, 5
}))
```

:::tip

For more advanced creation patterns, see [Operators](./operators) documentation.

:::

### From Custom Logic

:::warning Advanced Usage

Build custom observables with `ro.NewObservable()` when you need complete control over emission logic. This approach is useful for wrapping existing APIs or creating complex data sources.

:::

```go
// Create an Observable with custom emission logic
func createCounter() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 1; i <= 5; i++ {
            time.Sleep(5*time.Second)
            // The following line will block until all downstream operators receive
            // the message. This behavior is different from Go channels since producer
            // get released as soon as consumer read the channel.
            observer.Next(i)
        }
        observer.Complete()
        return nil // No cleanup needed
    })
}

counter := createCounter()
counter.Subscribe(ro.OnNext(func(n int) {
    fmt.Println(n) // 1, 2, 3, 4, 5
}))
```

:::danger Resource Management

Always return a cleanup function when creating custom observables that use resources like files, network connections, or goroutines. See [Subscription](./subscription) for proper resource management.

:::

### From Time-based Operations

Use time-based operators for periodic emissions or delayed execution. These operators return non-blocking observables that emit values asynchronously.

```go
// Create an Observable that emits values periodically
interval := ro.Interval(1 * time.Second)
interval.Subscribe(ro.OnNext(func(tick int64) {
    fmt.Println("Tick:", tick) // 0, 1, 2, 3, ... every second
}))

// Create an Observable that emits once after a delay
timer := ro.Timer(2 * time.Second)
timer.Subscribe(ro.OnNext(func(duration time.Duration) {
    fmt.Println("Timer fired after:", duration) // 2s
}))
```

### From Slices

Convert existing Go slices into observables using `ro.FromSlice()`. This is convenient when you already have data in a slice and want to process it reactively.

```go
// Convert a slice to an Observable
data := []string{"apple", "banana", "cherry"}
observable := ro.FromSlice(data)

observable.Subscribe(ro.OnNext(func(item string) {
    fmt.Println(item) // "apple", "banana", "cherry"
}))
```

## Subscribing to Observables

### Basic Subscription

Use `ro.OnNext()` for simple subscriptions when you only need to handle values. Note that this approach ignores errors and completion signals.

```go
observable := ro.Just(1, 2, 3)

// Simple subscription - only handle Next values
observable.Subscribe(ro.OnNext(func(value int) {
    fmt.Println("Received:", value)
}))
// Output: Received: 1, Received: 2, Received: 3
```

:::warning Error Handling

For production code, always use full observers with proper error handling. See [Observer](./observer) for complete documentation.

:::

### Full Observer

Use `ro.NewObserver()` for complete control over all notification types. This is recommended for production code where proper error handling is essential.

```go
observable := ro.Just(1, 2, 3)

// Complete observer with all callbacks
observable.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Println("Next:", value)
    },
    func(err error) {
        fmt.Println("Error:", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
// Output:
// Next: 1
// Next: 2
// Next: 3
// Completed
```

### Context-aware Subscription

Use context-aware subscriptions when you need timeout control or cancellation. The context is passed through the entire pipeline and can be used to stop processing.

```go
ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
defer cancel()

observable := ro.Interval(1 * time.Second)

subscription := observable.SubscribeWithContext(ctx, ro.NewObserverWithContext(
    func(ctx context.Context, value int64) {
        fmt.Println("Received:", value)
    },
    func(ctx context.Context, err error) {
        fmt.Println("Error:", err)
    },
    func(ctx context.Context) {
        fmt.Println("Completed")
    },
))

// Wait for completion or context cancellation
subscription.Wait()
```

## Observable Characteristics

### Cold Observables (Default)

By default, Observables are **cold**, meaning:
- Each subscription creates a new independent execution
- Values are produced fresh for each subscriber
- The Observable doesn't start emitting until subscribed

```go
source := ro.Just(1, 2, 3)

// Each subscriber gets independent execution
source.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Subscriber 1:", n)
}))

// It will subscribe sequentially, after the first subscription ends
source.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Subscriber 2:", n)
}))

// Output:
// Subscriber 1: 1
// Subscriber 1: 2
// Subscriber 1: 3
// Subscriber 2: 1
// Subscriber 2: 2
// Subscriber 2: 3
```

### Hot Observables

Hot Observables share a single execution across multiple subscribers:

```go
// Convert a cold Observable into a hot Observable
hot := ro.Connectable(ro.Just(1, 2, 3))

// Multiple subscribers share the same execution
sub1 := hot.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Hot 1:", n)
}))

sub2 := hot.Subscribe(ro.OnNext(func(n int) {
    fmt.Println("Hot 2:", n)
}))

// Start the Observable
connection := hot.Connect()
```

## Resource Management

### Subscriptions

Every subscription returns a `Subscription` object that can be used to manage the execution:

```go
observable := ro.Interval(1 * time.Second)

// Subscribe and get the subscription
subscription := observable.Subscribe(ro.OnNext(func(tick int64) {
    fmt.Println("Tick:", tick)
}))

// Since ro.Interval is async, .Subscribe(...) will be non-blocking.

// Cancel the subscription after 3 seconds
time.AfterFunc(3*time.Second, func() {
    subscription.Unsubscribe()
    fmt.Println("Unsubscribed")
})
```

### Cleanup with Teardown

Observables can return cleanup functions that are called when unsubscribed. This is essential for managing resources like files:

```go
func createFileObservable(filePath string) ro.Observable[string] {
    return ro.NewObservable(func(observer ro.Observer[string]) ro.Teardown {
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

        return func() {
            close(done)
            file.Close()
        }
    })
}
```

## Collecting Values

### Blocking Collection

Blocking behavior are discouraged in Reactive Programming. Use it carefuly.

```go
observable := ro.Just(1, 2, 3, 4, 5)

// Collect all values (blocks until completion)
values, err := ro.Collect(observable)
if err != nil {
    fmt.Println("Error:", err)
} else {
    fmt.Println("Values:", values) // [1, 2, 3, 4, 5]
}
```

### Context-aware Collection

```go
ctx := context.Background()
observable := ro.Just(1, 2, 3)

// Collect with context
values, lastCtx, err := ro.CollectWithContext(ctx, observable)
fmt.Println("Values:", values)      // [1, 2, 3]
fmt.Println("Context:", lastCtx)     // Final context state
fmt.Println("Error:", err)          // nil
```

## Error Handling

Observables can emit errors that terminate the stream:

```go
// Create an Observable that might error
func riskyObservable() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 1; i <= 5; i++ {
            if i == 3 {
                observer.Error(fmt.Errorf("error at %d", i))
                return nil
            }
            observer.Next(i)
        }
        observer.Complete()
        return nil
    })
}

riskyObservable().Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Println("Received:", value)
    },
    func(err error) {
        fmt.Println("Error:", err) // Error: error at 3
    },
    func() {
        fmt.Println("Completed")
    },
))
// Output:
// Received: 1
// Received: 2
// Error: error at 3
```

## Observable vs Traditional Iteration

**Pull-based (Traditional)**
```go
// Traditional iteration - consumer pulls values
numbers := []int{1, 2, 3, 4, 5}
for _, n := range numbers {
    fmt.Println(n) // Consumer controls when to get next value
}
```

**Push-based (Observable)**
```go
// Observable - producer pushes values
observable := ro.Just(1, 2, 3, 4, 5)
observable.Subscribe(ro.OnNext(func(n int) {
    fmt.Println(n) // Producer pushes values when ready
}))
```

## Blocking vs Non-blocking Subscriptions

An important distinction in `ro` is understanding when subscription calls block and when they return immediately:

### Understanding the Behavior

- **Blocking observables**: Complete synchronously and return a closed subscription
- **Non-blocking observables**: Return immediately and emit values asynchronously

### Blocking Subscription Example

Most of operators and creation operator are synchronous (eg: `ro.Just`)

```go
observable := ro.Just(1, 2, 3)

// This call blocks until all values are emitted
subscription := observable.Subscribe(ro.OnNext(func(value int) {
    fmt.Println("Received:", value)
}))
// Output:
// Received: 1
// Received: 2
// Received: 3

// The subscription is already closed when returned
fmt.Println(subscription.IsClosed())
// Output:
// true
```

### Non-blocking Subscription Example

Some creation operators like `ro.Interval` and a few operators like `ro.ObserveOn` have an async behaviour.

```go
observable := ro.Interval(1 * time.Second)

// This call returns immediately
subscription := observable.Subscribe(ro.OnNext(func(value int64) {
    fmt.Println("Tick:", value)
}))

// The subscription is still active
fmt.Println(subscription.IsClosed())
// Output:
// false

// Values will be emitted asynchronously
// Output:
// Subscription is active
// Tick: 0 (after 1 second)
// Tick: 1 (after 2 seconds)
// ...

time.Sleep(10*time.Second)
subscription.Unsubscribe()
```

### When to Expect Blocking Behavior

These observables typically **block** during subscription:
- `ro.Just()` - emits finite values immediately
- `ro.Range()` - emits finite sequence immediately
- `ro.FromSlice()` - emits slice contents immediately
- `ro.Empty()` - completes immediately
- `ro.Throw()` - errors immediately

These observables are typically **non-blocking**:
- `ro.Interval()` - emits values periodically
- `ro.Timer()` - emits once after delay
- Custom observables with async logic
- `ro.NewSubject()` - hot observables that emit when values are pushed

### Practical Implications

```go
func processData() {
    // This will block until all processing is complete
    data := ro.Just(1, 2, 3)
    data.Subscribe(ro.OnNext(func(n int) {
        fmt.Println("Processing:", n)
    }))
    fmt.Println("This line runs after processing completes")
}

func handleEvents() {
    // This returns immediately, allowing concurrent processing
    events := ro.Interval(1 * time.Second)
    subscription := events.Subscribe(ro.OnNext(func(tick int64) {
        fmt.Println("Event:", tick)
    }))
    fmt.Println("This line runs immediately")

    // Continue with other work while events stream in
    time.Sleep(3 * time.Second)
    subscription.Unsubscribe()
}
```

## Best Practices

### 1. Always Handle Cleanup

```go
// Good: Clean up resources
subscription := intervalObservable.Subscribe(observer)
defer subscription.Unsubscribe()

// Bad: Potential resource leak
intervalObservable.Subscribe(observer) // No cleanup
```

### 2. Use Context for Cancellation

```go
// Good: Use context for timeout/cancellation
ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
defer cancel()

subscription := observable.SubscribeWithContext(ctx, observer)
```

### 3. Handle Errors Gracefully

```go
// Good: Handle errors in observer
observable.Subscribe(ro.NewObserver(
    func(value int) { /* handle value */ },
    func(err error) { /* handle error */ },
    func() { /* handle completion */ },
))

// Bad: No error handling
observable.Subscribe(ro.OnNext(func(value int) { /* only handle values */ }))
```

### 4. Avoid Blocking Operations

```go
// Good: Non-blocking subscription
subscription := observable.Subscribe(observer)
// Continue with other work

// Questionable: Blocking wait
subscription.Wait() // Against reactive principles
```

## When to Use Observables

Observables excel in scenarios involving:
- **Event streams**: User interactions, network events, system events
- **Asynchronous operations**: API calls, database queries, file I/O
- **Real-time data**: Sensor readings, stock prices, chat messages
- **Time-based operations**: Timers, intervals, animations
- **Complex data pipelines**: Multi-step processing with error handling

Observables provide a powerful, declarative way to handle asynchronous data streams in Go, making complex event-driven applications more manageable and maintainable.
