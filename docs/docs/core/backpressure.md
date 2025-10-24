---
title: Backpressure
description: Discover how samber/ro handles backpressure.
sidebar_position: 6
---

# 🫷 Backpressure

Backpressure is a fundamental concept in reactive programming that handles the scenario where a producer emits data faster than a consumer can process it. In `samber/ro`, backpressure is handled naturally through the library's blocking behavior design.

## What is Backpressure?

Backpressure occurs when there's an imbalance between:
- **Producer rate**: How fast data is emitted
- **Consumer rate**: How fast data is processed

Without proper backpressure handling, systems can experience:
- Memory overflow from buffered messages
- System instability from resource exhaustion
- Lost messages when buffers overflow

## How samber/ro Handles Backpressure

### Natural Backpressure Through Blocking

`samber/ro` implements backpressure naturally through **blocking behavior**. When you call `observer.Next()`, the call blocks until all downstream operators in the pipeline have processed the value.

```go
func createFastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 1; i <= 5; i++ {
            fmt.Printf("Emitting %d\n", i)

            // This blocks until downstream completes
            observer.Next(i)

            fmt.Printf("Downstream completed for %d\n", i)
        }
        observer.Complete()
        return nil
    })
}

// Create a pipeline with a slow consumer
var pipeline = ro.Pipe1(
    createFastProducer(),
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

### Benefits of Blocking Backpressure

This natural backpressure approach provides several advantages:

1. **Bounded Memory**: No unbounded buffers accumulating messages
2. **No Message Loss**: Values are processed in order, none are dropped
3. **Flow Regulation**: Pipeline naturally regulates flow rate
4. **Sequential Processing**: Stream is consumed sequentially
5. **Message Order**: Order is preserved throughout the pipeline

## Backpressure in Different Scenarios

### Fast Producer, Slow Consumer

```go
// Fast producer (emits every 10ms)
fastProducer := ro.Interval(10 * time.Millisecond)

// Slow consumer (processes every 100ms)
pipeline := ro.Pipe1(
    fastProducer,
    ro.Map(func(value int64) int64 {
        time.Sleep(100 * time.Millisecond) // Slow processing
        return value * 2
    }),
)

// The producer will naturally wait for the consumer
subscription := pipeline.Subscribe(ro.OnNext(func(value int64) {
    fmt.Printf("Processed: %d\n", value)
}))

// Producer rate automatically adjusts to consumer rate
time.Sleep(1 * time.Second)
subscription.Unsubscribe()
```

### Multiple Subscribers

```go
source := ro.Pipe1(
    ro.Just(1, 2, 3, 4, 5),
    ro.Map(func(value int) int {
        // Simulate processing time
        time.Sleep(50 * time.Millisecond)
        return value * 2
    }),
)

// Multiple subscribers each get independent processing
sub1 := source.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Subscriber 1: %d\n", value)
}))

sub2 := source.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Subscriber 2: %d\n", value)
}))
```

## Custom Operators and Backpressure

When creating custom operators, you need to be mindful of backpressure:

### Good: Blocking Operator

```go
func SlowMap[T, R any](mapper func(T) R) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewUnsafeObservable(func(destination ro.Observer[R]) ro.Teardown {
            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    // This blocks until downstream can receive
                    result := mapper(value)
                    destination.Next(result)
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}
```

### Warning: Async Operators

If you create operators that emit asynchronously, be aware that you lose natural backpressure:

```go
// ⚠️ This operator breaks natural backpressure
func AsyncMap[T, R any](mapper func(T) R) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewSafeObservable(func(destination ro.Observer[R]) ro.Teardown {
            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    go func() {
                        // This doesn't block - can cause backpressure issues
                        result := mapper(value)
                        destination.Next(result)
                    }()
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}
```

## Backpressure vs Traditional Go Channels

### Go Channels (Manual Backpressure)

```go
// Traditional Go approach requires careful buffer management
ch := make(chan int, 100) // Fixed buffer size

go func() {
    for i := 0; i < 1000; i++ {
        select {
        case ch <- i:
            // OK - channel accepted value
        default:
            // Channel full - handle backpressure
            fmt.Println("Channel full, dropping value:", i)
            // Could block, drop, or use other strategy
        }
    }
    close(ch)
}()

for value := range ch {
    fmt.Println("Received:", value)
    time.Sleep(10 * time.Millisecond) // Slow processing
}
```

### samber/ro (Natural Backpressure)

```go
// ro handles backpressure automatically
observable := ro.Range(0, 1000)
pipeline := ro.Pipe1(
    observable,
    ro.Map(func(value int) int {
        // Producer naturally waits for slow consumer
        return value * 2
    }),
)

pipeline.Subscribe(ro.OnNext(func(value int) {
    fmt.Println("Received:", value)
    time.Sleep(10 * time.Millisecond) // Slow processing
}))
```

## Practical Considerations

### Memory Usage

Because `samber/ro` uses blocking behavior, memory usage remains bounded:

```go
// This won't consume unbounded memory
pipeline := ro.Pipe1(
    ro.Interval(1 * time.Millisecond),  // Fast producer
    ro.Map(func(value int64) int64 {
        time.Sleep(100 * time.Millisecond) // Slow consumer
        return value
    }),
)

// Memory usage stays constant regardless of runtime
subscription := pipeline.Subscribe(ro.OnNext(func(value int64) {
    fmt.Println("Value:", value)
}))
```

### Error Handling with Backpressure

Error handling works seamlessly with backpressure:

```go
func riskyProcessing(value int) (int, error) {
    if value == 5 {
        return 0, fmt.Errorf("processing failed for value %d", value)
    }
    return value * 2, nil
}

pipeline := ro.Pipe1(
    ro.Just(1, 2, 3, 4, 5, 6),
    ro.MapErr(riskyProcessing),
)

pipeline.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Println("Success:", value)
    },
    func(err error) {
        fmt.Println("Error:", err) // Processing stops here
    },
    func() {
        fmt.Println("Completed")
    },
))
// Output:
// Success: 2
// Success: 4
// Success: 6
// Success: 8
// Error: processing failed for value 5
```

### Timeouts and Backpressure

```go
ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
defer cancel()

pipeline := ro.Pipe1(
    ro.Interval(100 * time.Millisecond),
    ro.Map(func(value int64) int64 {
        time.Sleep(200 * time.Millisecond) // Slow processing
        return value
    }),
)

pipeline.SubscribeWithContext(ctx, ro.NewObserverWithContext(
    func(ctx context.Context, value int64) {
        fmt.Println("Value:", value)
    },
    func(ctx context.Context, err error) {
        fmt.Println("Error:", err) // Timeout after 500ms
    },
    func(ctx context.Context) {
        fmt.Println("Completed")
    },
))
```

## Observable Types and Backpressure

`samber/ro` provides three types of observables with different backpressure characteristics:

### Unsafe Observables (Should be used for Synchronous Operations)

Unsafe observables are the fastest option but provide no protection against concurrent message passing. They are ideal when your operators are purely synchronous.

```go
// Fastest option for synchronous operations
func FastMap[T, R any](mapper func(T) R) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewUnsafeObservable(func(destination ro.Observer[R]) ro.Teardown {
            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    // No synchronization overhead
                    result := mapper(value)
                    destination.Next(result) // Blocks naturally for backpressure
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}
```

**Backpressure behavior**: Natural blocking through `destination.Next()`

**Use when**:
- All operations are synchronous
- Maximum performance is required
- No concurrent access to observers

### Safe Observables (Default)

Safe observables prevent concurrent message passing through the observer, ensuring thread safety at the cost of some performance overhead.

```go
// Thread-safe option for potentially concurrent operations
func SafeMap[T, R any](mapper func(T) R) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewSafeObservable(func(destination ro.Observer[R]) ro.Teardown {
            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    result := mapper(value)

                    // Synchronization prevents race conditions

                    go func() {
                        destination.Next(result) // Thread-safe backpressure
                    }()

                    go func() {
                        destination.Next(result*2) // Thread-safe backpressure
                    }()
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}
```

**Backpressure behavior**: Natural blocking with thread synchronization

**Use when**:
- Operations might be concurrent
- Thread safety is required
- Moderate performance is acceptable

### Eventually Safe Observables (Drop Strategy)

Eventually safe observables handle concurrency by dropping concurrent messages instead of blocking. This provides a different backpressure strategy.

```go
// Drop strategy for high-throughput scenarios
func HighThroughputMap[T, R any](mapper func(T) R) func(ro.Observable[T]) ro.Observable[R] {
    return func(source ro.Observable[T]) ro.Observable[R] {
        return ro.NewEventuallySafeObservable(func(destination ro.Observer[R]) ro.Teardown {
            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    result := mapper(value)
                    // May drop concurrent messages instead of blocking
                    destination.Next(result)
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}
```

**Backpressure behavior**: Can drop messages instead of blocking

**Use when**:
- Message loss is acceptable
- High throughput is prioritized over message delivery
- You want a "lossy" backpressure strategy

### Comparison Table

| Observable Type     | Thread Safety | Performance | Backpressure Strategy    | Message Loss |
| ------------------- | ------------- | ----------- | ------------------------ | ------------ |
| **Unsafe**          | No            | Highest     | Natural blocking         | No           |
| **Safe**            | Yes           | High        | Natural blocking + sync  | No           |
| **Eventually Safe** | Yes           | Medium      | Drop concurrent messages | Yes          |

The choice of observable type directly impacts how backpressure is handled:

1. **Unsafe**: Producer blocks until consumer is ready (perfect backpressure)
2. **Safe**: Producer blocks with synchronization overhead (perfect backpressure + thread safety)
3. **Eventually Safe**: Producer may drop messages instead of blocking (lossy backpressure)

## Serializing Observable Streams

The `ro.Serialize()` operator ensures thread-safe message passing by wrapping any observable in a safe observable implementation. This is useful when you need guaranteed serialization in concurrent scenarios.

```go
// Async concurrent producer that emits from multiple goroutines
func createConcurrentProducer() ro.Observable[int] {
    return ro.NewUnsafeObservable(func(observer ro.Observer[int]) Teardown {
        for i := 0; i < 3; i++ {
            go func(id int) {
                for j := 0; j < 5; j++ {
                    value := id*10 + j
                    observer.Next(value) // Concurrent emissions
                }
            }(i)
        }
        observer.Complete()
        return nil
    })
}

// Serialize ensures thread-safe message passing
pipeline := ro.Pipe2(
    createConcurrentProducer(),
    ro.Serialize[int](), // Wraps in safe observable for serialization
    ro.Distinct[int](),  // Distinct operator is not protected against race conditions
)

subscription := pipeline.Subscribe(ro.OnNext(func(value int) {
    fmt.Printf("Received: %d\n", value)
}))
```

**Backpressure behavior**: Same as Safe observables - natural blocking with synchronization overhead that ensures sequential message processing.

## Buffering with ObserveOn and SubscribeOn

`samber/ro` provides scheduling operators that use buffered channels for backpressure:

```go
// Both operators create buffered channels
ro.SubscribeOn[int](10)  // Buffer of 10 items
ro.ObserveOn[int](10)     // Buffer of 10 items
```

### Buffer Size and Backpressure

The buffer size directly controls backpressure behavior:

- **Small buffer (1-100)**: Tight backpressure, upstream blocks frequently
- **Medium buffer (100-1_000)**: Balanced, good for most use cases
- **Large buffer (1_000+)**: Loose backpressure, allows more buffering

**How it works**:
1. Creates buffered channel: `make(chan Notification, bufferSize)`
2. **Upstream blocks** when buffer is full
3. **Flow regulation** prevents memory overflow
4. **FIFO order** maintains message sequence

```go
// Example: Fast producer with small buffer
pipeline := ro.Pipe1(
    ro.Interval(1 * time.Millisecond),  // Fast producer
    ro.ObserveOn[int64](5),             // Small buffer
)

// Upstream will block when buffer of 5 is full
// Natural backpressure regulates flow rate
```

## Buffering with ro.Buffer and variants

The `BufferWithTimeOrCount` operator helps manage backpressure by collecting items into batches with both size and time limits:

```go
// Buffer every 3 items OR every 100ms, whichever comes first
pipeline := ro.Pipe3(
    ro.Just(1, 2, 3, 4, 5, 6),
    ro.BufferWithTimeOrCount[int](3, 100 * time.Millisecond),
    ro.Map(func(batch []int) []Result {
        return batchProcessing(batch)
    }),
    ro.Flatten[Result](),
)
```

**Backpressure benefits**:
- **Adaptive batching**: Handles both high-frequency and sparse data
- **Memory safety**: Never exceeds maximum batch size (3 items)
- **Responsive**: Emits batches even if source is slow (100ms timeout)
- **Flow regulation**: Reduces number of items sent downstream

This operator provides flexible backpressure control by ensuring buffers are released based on either volume or time, preventing both memory buildup and excessive delays.

## Advanced Backpressure Patterns

### Batching for Efficiency

While `samber/ro` handles individual item backpressure, you can implement batching for efficiency:

```go
func BatchProcessor[T any](batchSize int, processor func([]T)) func(ro.Observable[T]) ro.Observable[[]T] {
    return func(source ro.Observable[T]) ro.Observable[[]T] {
        return ro.NewUnsafeObservable(func(destination ro.Observer[[]T]) ro.Teardown {
            batch := make([]T, 0, batchSize)

            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    batch = append(batch, value)
                    if len(batch) >= batchSize {
                        processor(batch)
                        destination.Next(append([]T{}, batch...))
                        batch = batch[:0] // Reset slice
                    }
                },
                destination.Error,
                func() {
                    if len(batch) > 0 {
                        processor(batch)
                        destination.Next(batch)
                    }
                    destination.Complete()
                },
            ))
        })
    }
}

// Usage
var pipeline = ro.Pipe1(
    ro.Just(1, 2, 3, 4, 5, 6, 7),
    BatchProcessor(3, func(batch []int) {
        fmt.Printf("Processing batch: %v\n", batch)
    }),
)

pipeline.Subscribe(ro.OnNext(func(batch []int) {
    fmt.Printf("Received batch: %v\n", batch)
}))
```

### Throttling

Implement throttling to limit processing rate:

```go
func Throttle[T any](interval time.Duration) func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewUnsafeObservable(func(destination ro.Observer[T]) ro.Teardown {
            lastEmit := time.Now()

            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    now := time.Now()
                    if now.Sub(lastEmit) >= interval {
                        destination.Next(value)
                        lastEmit = now
                    }
                    // If not enough time has passed, the value is dropped
                    // This provides a form of backpressure through dropping
                },
                destination.Error,
                destination.Complete,
            ))
        })
    }
}

// Usage
pipeline := ro.Pipe2(
    ro.Interval(10 * time.Millisecond),  // Fast producer
    Throttle(100 * time.Millisecond),   // Limit to 10 per second
)

pipeline.Subscribe(ro.OnNext(func(value int64) {
    fmt.Println("Throttled value:", value)
}))
```

## Best Practices

### 1. Trust Natural Backpressure

```go
// Good: Let ro handle backpressure naturally
pipeline := ro.Pipe1(
    fastProducer,
    slowOperator,
)

pipeline.Subscribe(observer)
```

### 2. Avoid Bypassing Backpressure

```go
// Bad: This breaks natural backpressure
pipeline := ro.Pipe1(
    source,
    ro.Map(func(value int) int {
        go func() {
            // Async processing bypasses backpressure
            result := slowOperation(value)
            // Where does result go? No backpressure control
        }()
        return value
    }),
)

// Good: Keep processing synchronous in the pipeline
pipeline := ro.Pipe1(
    source,
    ro.Map(func(value int) int {
        // This blocks naturally, providing backpressure
        return slowOperation(value)
    }),
)
```

### 3. Handle Resource Cleanup

```go
func ResourceIntensiveOperator[T any]() func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewUnsafeObservable(func(destination ro.Observer[T]) ro.Teardown {
            // Acquire resource
            resource := acquireExpensiveResource()

            return source.Subscribe(ro.NewObserver(
                func(value T) {
                    // Process with resource
                    result := processWithResource(resource, value)
                    destination.Next(result)
                },
                destination.Error,
                destination.Complete,
            ))

            // Cleanup function called on unsubscription/completion/error
            return func() {
                releaseExpensiveResource(resource)
            }
        })
    }
}
```

### 4. Monitor Performance

```go
// Add timing to understand backpressure behavior
pipeline := ro.Pipe2(
    source,
    ro.Map(func(value int) int {
        start := time.Now()
        result := expensiveOperation(value)
        duration := time.Since(start)

        if duration > 100*time.Millisecond {
            fmt.Printf("Slow operation took %v for value %d\n", duration, value)
        }

        return result
    }),
)
```

## When Backpressure Matters Most

Backpressure is particularly important in these scenarios:

1. **File Processing**: Large files processed line by line
2. **Network Streams**: High-volume network data processing
3. **Database Operations**: Batch processing of database records
4. **API Integration**: Rate-limited external API calls
5. **Real-time Analytics**: Processing sensor data or metrics
6. **Image/Video Processing**: Heavy computational operations

## Summary

`samber/ro` provides a simple yet powerful approach to backpressure through natural blocking behavior:

- **Automatic**: No need for explicit backpressure handling
- **Safe**: Bounded memory usage prevents resource exhaustion
- **Predictable**: Sequential processing maintains order
- **Simple**: Easy to understand and reason about
- **Flexible**: Multiple observable types for different use cases

The choice of observable type (unsafe, safe, or eventually safe) directly impacts backpressure behavior.

This design choice makes `samber/ro` particularly well-suited for applications where reliability and predictable resource usage are more important than maximum throughput, while still providing options for different performance and concurrency requirements.

---

**Related Topics:**
- [Observable](./observable) - Understanding data producers
- [Observer](./observer) - Understanding data consumers
- [Operators](../operator) - Data transformation operations
- [Subject](./subject) - Hot observables and multicasting
