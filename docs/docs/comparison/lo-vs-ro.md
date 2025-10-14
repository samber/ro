---
title: samber/lo vs samber/ro
description: Compare samber/lo (functional utils) vs samber/ro (reactive streams)
sidebar_position: 5
---

# ⚖️ `samber/lo` vs `samber/ro`

Both `samber/lo` and `samber/ro` are powerful Go libraries, but they serve different purposes:

- **samber/lo**: A Lodash-like utility library for Go (bounded slices)
- **samber/ro**: A Reactive Programming library for Go (unbounded and event-driven streams)

This comparison will help you understand when to use each library and how they can complement each other.

## Key Differences

:::tip Core Distinctions

### Paradigm
- **lo**: **Synchronous** functional programming
- **ro**: **Asynchronous** reactive programming

### Data Flow
- **lo**: Immediate computation on finite collections
- **ro**: Stream processing on potentially infinite data sources

### Use Cases
- **lo**: Data transformation, validation, filtering on existing data
- **ro**: Event handling, real-time processing, async workflows

:::

The fundamental difference lies in how each library handles data flow and execution timing.

## Code Comparison

### Data Transformation

**samber/lo** (synchronous):
```go
package main

import (
    "fmt"
    "github.com/samber/lo"
)

func main() {
    numbers := []int{1, 2, 3, 4, 5}

    stage1 := lo.Filter(numbers, func(x int) bool {
        return x%2==0
    })
    stage2 := lo.Map(stage1, func(x int, _ int) string {
        return fmt.Sprintf("num-%d", x)
    })

    fmt.Println(stage2) // ["num-1", "num-2", "num-3", "num-4", "num-5"]
}
```

:::warning Stream Processing

**samber/ro**:
```go
package main

import (
    "fmt"
    "github.com/samber/ro"
)

func main() {
    observable := ro.Pipe2(
        ro.Just(1, 2, 3, 4, 5),
        ro.Filter(func(x int) bool {
            return x%2==0
        }),
        ro.Map(func(x int) string {
            return fmt.Sprintf("num-%d", x)
        }),
    )

    observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s) // "num-2", "num-4"
    }))
}
```

:::

Notice how `ro` processes values as a stream, while `lo` processes the entire collection at once.

### Filtering

:::tip Immediate Results

**samber/lo**:

:::

Results are available immediately after the function call.
```go
numbers := []int{1, 2, 3, 4, 5}
evens := lo.Filter(numbers, func(x int, _ int) bool {
    return x%2 == 0
})
// evens = [2, 4]
```

**samber/ro**:
```go
observable := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5),
    ro.Filter(func(x int) bool {
        return x%2 == 0
    }),
)

observable.Subscribe(ro.OnNext(func(x int) {
    fmt.Println(x) // 2, 4
}))
```

:::

Filtering happens as values flow through the stream, providing lazy evaluation.

### Async vs Sync

:::danger Blocking Behavior

**samber/lo** (blocking):

:::

All processing must complete before the function returns, blocking execution.
```go
func processData(data []int) []string {
    // Blocks until all processing is complete
    return lo.Map(
        lo.Filter(data, func() bool {
            return i%2 == 1
        }),
        func(x int, _ int) string {
            time.Sleep(100 * time.Millisecond) // blocking
            return fmt.Sprintf("processed-%d", x)
        },
    )
}

func main() {
    // Synchronous call
    result := processData([]int{1, 2, 3})
    fmt.Println(result) // appears after 200ms
}
```

:::tip Non-blocking Streams

**samber/ro** (non-blocking):

:::

Values are processed as they arrive, without blocking the main execution flow.
```go
var pipeline = ro.PipeOp3(
    ro.Filter(func(x int) bool {
        return x%2 == 1
    })
    ro.Map(func(x int) string {
        return fmt.Sprintf("processed-%d", x)
    }),
    ro.DelayEach[string](100 * time.Millisecond)
)

func main() {
    observable := pipeline(ro.Just(1, 2, 3))

    // Non-blocking subscription
    _ = observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s) // appears immediately, one by one
    }))
}
```

## When to Use Which

:::info Decision Guide

### Use samber/lo when:
- Working with existing data collections
- Need immediate, synchronous results
- Performing data validation and transformation
- Writing utility functions and helpers
- Need comprehensive functional programming utilities

### Use samber/ro when:
- Handling real-time or external events (clicks, websockets, timers)
- Working with infinite data sources
- Processing streaming data
- Building reactive user interfaces
- Implementing async workflows
- Need backpressure handling

:::

Consider your specific use case requirements when choosing between these libraries.

## Combining Both Libraries

:::tip Best of Both Worlds

You can use both libraries together for maximum flexibility:

:::

Use `lo` for data preparation and `ro` for stream processing - they complement each other perfectly.

```go
package main

import (
    "fmt"
    "github.com/samber/lo"
    "github.com/samber/ro"
)

func main() {
    // Use lo for initial data preparation
    numbers := lo.Range(1, 11)
    evens := lo.Filter(numbers, func(x int, _ int) bool {
        return x%2 == 0
    })

    // Use ro for real-time processing
    observable := ro.Pipe2(
        ro.Just(evens...),
        ro.Map(func(x int) string {
            return fmt.Sprintf("stream-%d", x)
        }),
    )

    observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s)
    }))
}
```

## Performance Characteristics

:::warning Performance Considerations

| Aspect           | samber/lo                       | samber/ro               |
| ---------------- | ------------------------------- | ----------------------- |
| **Memory Usage** | Higher (accumulate collections) | Lower (lazy producing)  |
| **Latency**      | Low (blocks until complete)     | medium (small overhead) |
| **CPU Usage**    | Predictable                     | Predictable             |
| **Concurrency**  | None                            | Built-in                |
| **Backpressure** | Not applicable                  | Automatic               |

:::

Choose based on your specific performance requirements - `lo` for immediate results, `ro` for streaming efficiency.

## Feature Comparison

:::info Feature Matrix

| Feature               | samber/lo | samber/ro |
| --------------------- | --------- | --------- |
| Map/Filter            | ✅         | ✅         |
| Reduce/Fold           | ✅         | ✅         |
| Async Processing      | ❌         | ✅         |
| Error Handling        | Basic     | Advanced  |
| Retry Mechanisms      | ❌         | ✅         |
| Time-based Operations | ❌         | ✅         |
| Backpressure          | ❌         | ✅         |
| Hot/Cold Observables  | ❌         | ✅         |
| Subject Types         | ❌         | ✅         |

:::

Both libraries excel in their respective domains. Choose `lo` for traditional functional programming on collections and `ro` for reactive, event-driven applications.

:::tip Learn More

- Explore [samber/ro basics](./core/observable) for reactive concepts
- See [Operators guide](./core/operators) for stream transformations
- Learn about [backpressure](./glossary#Backpressure) in reactive systems

:::
