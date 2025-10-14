---
title: 📡 channels vs ro
description: Compare Go channels vs samber/ro reactive streams
sidebar_position: 3
---

# 📡 `channels` vs `samber/ro`

Go's built-in channels and `samber/ro` both provide mechanisms for handling concurrent data flow, but they represent different programming paradigms:

- **Go channels**: Low-level concurrency primitives with explicit control
- **samber/ro**: High-level reactive streams with declarative operators

This comparison explores how these two approaches handle concurrency, data flow, and stream processing in Go.

## Key Differences

:::tip Core Distinctions

### Abstraction Level
- **channels**: **Low-level** building blocks for concurrency
- **samber/ro**: **High-level** declarative stream processing

### Communication Pattern
- **channels**: Point-to-point communication between goroutines
- **samber/ro**: Broadcast to multiple subscribers with automatic fan-out

### Error Handling
- **channels**: Manual error propagation (or separate error channels)
- **samber/ro**: Built-in error propagation and recovery mechanisms

:::

The fundamental difference lies in how each approach handles concurrency and data flow - channels provide explicit control while `ro` offers declarative stream processing.

## Code Comparison

### Basic Data Flow

**Go channels**:
```go
package main

import (
    "fmt"
)

func main() {
    // Create channel
    ch := make(chan int)

    // Producer goroutine
    go func() {
        for i := 1; i <= 5; i++ {
            ch <- i
        }
        close(ch)
    }()

    // Consumer
    for value := range ch {
        fmt.Println(value) // 1, 2, 3, 4, 5
    }
}
```

:::

Channels require manual goroutine management and explicit synchronization.

:::tip Declarative Streams

**samber/ro**:
```go
package main

import (
    "fmt"
    "github.com/samber/ro"
)

func main() {
    // Create observable stream
    observable := ro.Just(1, 2, 3, 4, 5)

    // Subscribe
    observable.Subscribe(ro.OnNext(func(value int) {
        fmt.Println(value) // 1, 2, 3, 4, 5
    }))
}
```

:::

Reactive streams provide automatic synchronization and declarative composition.

### Multiple Consumers

:::warning Manual Fan-out

**Go channels** (manual fan-out):
```go
func fanOut(ch <-chan int, outputs []chan<- int) {
    defer func() {
        for _, out := range outputs {
            close(out)
        }
    }()

    for value := range ch {
        for _, out := range outputs {
            out <- value
        }
    }
}

func main() {
    input := make(chan int)
    output1 := make(chan int)
    output2 := make(chan int)

    // Start fan-out goroutine
    go fanOut(input, []chan<- int{output1, output2})

    // Producer
    go func() {
        defer close(input)
        for i := 1; i <= 5; i++ {
            input <- i
        }
    }()

    // Consumers
    go func() {
        for v := range output1 {
            fmt.Println("Consumer 1:", v)
        }
    }()

    go func() {
        for v := range output2 {
            fmt.Println("Consumer 2:", v)
        }
    }()

    // Wait for completion
    time.Sleep(100 * time.Millisecond)
}
```

:::

With channels, you need to manually implement fan-out logic and manage multiple output channels.

**samber/ro** (implicit fan-out):
```go
func main() {
    // Single observable, multiple subscribers
    observable := ro.Just(1, 2, 3, 4, 5)

    // Multiple subscribers automatically get all values
    observable.Subscribe(ro.OnNext(func(v int) {
        fmt.Println("Subscriber 1:", v)
    }))

    observable.Subscribe(ro.OnNext(func(v int) {
        fmt.Println("Subscriber 2:", v)
    }))

    // No need to manage goroutines or channels
}
```

:::

Multiple subscribers automatically receive all values without manual channel management.

### Error Handling

:::danger Manual Error Management

**Go channels**:
```go
type Result struct {
    Value int
    Error error
}

func processData(data []int) <-chan Result {
    ch := make(chan Result)

    go func() {
        defer close(ch)

        for _, item := range data {
            result, err := processItem(item)
            ch <- Result{Value: result, Error: err}
        }
    }()

    return ch
}

func processItem(item int) (int, error) {
    if item == 3 {
        return 0, fmt.Errorf("error processing %d", item)
    }
    return item * 2, nil
}

func main() {
    results := processData([]int{1, 2, 3, 4, 5})

    for result := range results {
        if result.Error != nil {
            fmt.Printf("Error: %v\n", result.Error)
        } else {
            fmt.Printf("Value: %d\n", result.Value)
        }
    }
}
```

:::

Error handling requires custom structs and manual propagation through the pipeline.

:::tip Built-in Error Recovery

**samber/ro**:
```go
var pipeline = ro.PipeOp1(
    ro.MapErr(func(item int) (int, error) {
        if item == 3 {
            return 0, fmt.Errorf("error processing %d", item)
        }
        return item * 2, nil
    }),
)

func main() {
    observable := pipeline(ro.Just(1, 2, 3, 4, 5))

    observable.Subscribe(ro.NewObserver[int](
        func(value int) {
            fmt.Printf("Value: %d\n", value)
        },
        func(err error) {
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            fmt.Printf("Completed\n")
        },
    ))
}
```

:::

Error handling is built into the Observable pattern with dedicated error channels.

### Backpressure Handling

:::warning Manual Backpressure

**Go channels** (manual):
```go
func consumer(input <-chan int, output chan<- int) {
    defer close(output)

    for value := range input {
        // Simulate slow processing
        time.Sleep(100 * time.Millisecond)
        output <- value * 2
    }
}

func main() {
    // Buffered channel for some backpressure
    input := make(chan int, 10)
    output := make(chan int, 10)

    // Start consumer
    go consumer(input, output)

    // Producer with rate limiting
    go func() {
        defer close(input)
        for i := 1; i <= 20; i++ {
            select {
            case input <- i:
                // Successfully sent
            case <-time.After(50 * time.Millisecond):
                // Backpressure - consumer can't keep up
                fmt.Printf("Dropped value: %d\n", i)
            }
        }
    }()

    // Collect results
    for value := range output {
        fmt.Println("Processed:", value)
    }
}
```

:::

Backpressure requires manual implementation with buffered channels and timeout logic.

:::tip Automatic Flow Control

**samber/ro** (automatic):
```go
func main() {
    observable := ro.Pipe2(
        ro.Interval(50 * time.Millisecond),  // Fast producer
        ro.Take(20),
        ro.Map(func(i int) int {
            // Simulate slow processing
            time.Sleep(100 * time.Millisecond)
            return i * 2
        }),
    )

    observable.Subscribe(ro.OnNext(func(value int) {
        fmt.Println("Processed:", value)
    }))
}
```

:::

Backpressure is handled automatically through blocking `observer.Next()` calls.

### Complex Data Pipelines

:::danger Manual Pipeline Construction

**Go channels**:
```go
func stage1(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            output <- v * 2
        }
    }()
    return output
}

func stage2(input <-chan int) <-chan int {
    output := make(chan int)
    go func() {
        defer close(output)
        for v := range input {
            if v%4 == 0 {
                output <- v
            }
        }
    }()
    return output
}

func main() {
    source := make(chan int)

    // Build pipeline
    stage1Out := stage1(source)
    stage2Out := stage2(stage1Out)

    // Start producer
    go func() {
        defer close(source)
        for i := 0; i < 10; i++ {
            source <- i
        }
    }()

    // Collect results
    for result := range stage2Out {
        fmt.Println("Result:", result)
    }
}
```

:::

Complex pipelines require manual stage management and multiple goroutines.

**samber/ro**:
```go
func main() {
    // Declarative pipeline
    observable := ro.Pipe3(
        ro.Range(0, 10),
        ro.Map(func(x int) int {
            return x * 2
        }),
        ro.Filter(func(x int) bool {
            return x%4 == 0
        }),
    )

    observable.Subscribe(ro.OnNext(func(result int) {
        fmt.Println("Result:", result)
    }))
}
```

## Advanced Features Comparison

### Time-based Operations

**Go channels** (manual):
```go
func tickerStream(interval time.Duration) <-chan int {
    ch := make(chan int)

    go func() {
        defer close(ch)
        ticker := time.NewTicker(interval)
        defer ticker.Stop()

        counter := 0
        for range ticker.C {
            counter++
            select {
            case ch <- counter:
            default:
                // Handle backpressure
            }
        }
    }()

    return ch
}

func main() {
    ticks := tickerStream(time.Second)

    // Timeout after 5 seconds
    timeout := time.After(5 * time.Second)

    for {
        select {
        case tick := <-ticks:
            fmt.Println("Tick:", tick)
        case <-timeout:
            fmt.Println("Timeout")
            return
        }
    }
}
```

**samber/ro** (built-in):
```go
var pipeline = ro.PipeOp2(
    ro.Map(func(tick int) string {
        return fmt.Sprintf("tick-%d", tick)
    }),
    ro.TakeUntil(ro.Timer(5 * time.Second)),
)

func main() {
    observable := pipeline(ro.Interval(time.Second))

    observable.Subscribe(ro.OnNext(func(msg string) {
        fmt.Println(msg)
    }))
}
```

### Retry Mechanisms

**Go channels** (manual implementation):
```go
func withRetry(input <-chan int, maxRetries int) <-chan int {
    output := make(chan int)

    go func() {
        defer close(output)

        for value := range input {
            for attempt := 0; attempt <= maxRetries; attempt++ {
                result, err := processWithRetry(value, attempt)
                if err == nil {
                    output <- result
                    break
                }
                if attempt == maxRetries {
                    fmt.Printf("Failed after %d retries: %v\n", maxRetries, err)
                }
            }
        }
    }()

    return output
}

func main() {
    input := make(chan int)
    retryable := withRetry(input, 3)

    // Producer
    go func() {
        defer close(input)
        input <- 1
        input <- 2
        input <- 3
    }()

    // Consumer
    for result := range retryable {
        fmt.Println("Result:", result)
    }
}
```

**samber/ro** (built-in):
```go
var pipeline = ro.PipeOp2(
    ro.MapErr(func(x int) (int, error) {
        if x == 2 {
            return 0, fmt.Errorf("error for %d", x)
        }
        return x * 2, nil
    }),
    ro.Retry(3),
)

func main() {
    observable := pipeline(ro.Just(1, 2, 3))

    observable.Subscribe(ro.NewObserver[int](
        func(result int) {
            fmt.Println("Result:", result)
        },
        func(err error) {
            fmt.Println("Final error:", err)
        },
        func() {
            fmt.Println("Completed")
        },
    ))
}
```

## When to Use Which

:::info Decision Guide

### Use Go channels when:
- Need fine-grained control over goroutines
- Building low-level concurrent algorithms
- Performance is critical and overhead matters
- Working with existing channel-based code
- Simple point-to-point communication

### Use samber/ro when:
- Building complex data processing pipelines
- Need automatic backpressure handling
- Multiple subscribers required
- Want declarative, composable operators
- Need built-in error handling and retry mechanisms
- Time-based operations are needed

:::

Consider your specific requirements for control, complexity, and maintainability when choosing between these approaches.

## Performance Characteristics

:::warning Performance Considerations

| Aspect           | Go channels            | samber/ro                 |
| ---------------- | ---------------------- | ------------------------- |
| **Memory Usage** | Minimal                | Minimal                   |
| **Latency**      | Low                    | Very low                  |
| **CPU Usage**    | Minimal                | Moderate                  |
| **Control**      | Full control           | Abstracted away           |
| **Scalability**  | Manual scaling         | Automatic fan-out         |
| **Backpressure** | Unblock on consumption | Unblock after consumption |

:::

Channels are actually slower than sequential function chaining in `samber/ro`.

:::tip Backpressure Details

The key performance difference is in backpressure handling:
- **Channels**: Producer continues immediately after send, blocks only if buffer is full
- **ro**: Producer waits until consumer completes processing before continuing

Learn more about [backpressure](./glossary#Backpressure) in the glossary.

:::

## Feature Comparison

| Feature                      | Go channels | samber/ro |
| ---------------------------- | ----------- | --------- |
| Point-to-point Communication | ✅           | ✅         |
| Broadcast/Fan-out            | Manual      | ✅         |
| Error Handling               | Manual      | ✅         |
| Retry Mechanisms             | Manual      | ✅         |
| Time Operations              | Manual      | ✅         |
| Backpressure                 | Manual      | ✅         |
| Type Safety                  | ✅           | ✅         |
| Standard Library             | ✅           | ❌         |
| Goroutine Management         | Manual      | Automatic |
| Composition                  | Manual      | ✅         |

## Migration Examples

### From channels to ro

**Before (channels)**:
```go
func processStream(input <-chan int) <-chan string {
    output := make(chan string)

    go func() {
        defer close(output)

        for value := range input {
            processed := fmt.Sprintf("processed-%d", value)
            output <- processed
        }
    }()

    return output
}

func main() {
    source := make(chan int)
    processed := processStream(source)

    go func() {
        defer close(source)
        source <- 1
        source <- 2
        source <- 3
    }()

    for result := range processed {
        fmt.Println(result)
    }
}
```

**After (samber/ro)**:
```go
func main() {
    observable := ro.Pipe2(
        ro.Just(1, 2, 3),
        ro.Map(func(value int) string {
            return fmt.Sprintf("processed-%d", value)
        }),
    )

    observable.Subscribe(ro.OnNext(func(result string) {
        fmt.Println(result)
    }))
}
```

Go channels provide the foundation for concurrent programming in Go, while `samber/ro` builds upon these concepts to provide a higher-level, more declarative approach to stream processing. Choose channels for fine-grained control and `samber/ro` for expressive, maintainable stream processing.
