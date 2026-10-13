---
title: samber/lo and samber/ro
description: samber/lo and samber/ro are companion Go libraries by the same author — lo for synchronous collections, ro for asynchronous streams. Learn when to use each, and how to use both together.
sidebar_position: 5
---

# 🤝 `samber/lo` and `samber/ro`

**`samber/lo` and `samber/ro` solve different problems and are designed to work together.** `lo` is a Lodash-style toolkit for looping over finite, in-memory Go collections (slices, maps) synchronously. `ro` is a reactive streams library for processing asynchronous, potentially infinite sequences of events. Most Go services that consume `ro` for event-driven pipelines also use `lo` for the synchronous, in-memory steps around them — they are not alternatives to pick between, they're two tools for two different jobs.

- **samber/lo**: synchronous helpers for finite, in-memory collections (slices, maps)
- **samber/ro**: synchronous-by-default reactive streams for asynchronous, potentially infinite sequences of events

## Key Differences

:::tip Core Distinctions

### Paradigm
- **lo**: **Synchronous** functional programming
- **ro**: **Synchronous by default** reactive programming — no scheduler, reacts to events as they occur

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

    stage1 := lo.Filter(numbers, func(x int, _ int) bool {
        return x%2==0
    })
    stage2 := lo.Map(stage1, func(x int, _ int) string {
        return fmt.Sprintf("num-%d", x)
    })

    fmt.Println(stage2) // ["num-2", "num-4"]
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
observable := ro.Pipe[int, int](
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

### Both are synchronous by default — they differ in when results become visible

:::info Blocking vs. Progressive

- **lo**: computes the whole result before returning it — nothing is visible until the call returns
- **ro**: `Subscribe()` blocks the calling goroutine too, but surfaces each value to your observer as soon as it's produced, not all at once

:::

`ro` is **mostly synchronous and has no scheduler** (Go already gives you first-class concurrency, so `ro` doesn't invent its own — see the [glossary](../glossary#Asynchronous)). A default `Subscribe()` call blocks the calling goroutine exactly like a `lo` call blocks the calling function, until the pipeline completes.

**samber/lo** — the caller sees nothing until the whole slice is ready:
```go
func processData(data []int) []string {
    // Blocks until all processing is complete
    return lo.Map(
        lo.Filter(data, func(x int, _ int) bool {
            return x%2 == 1
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

**samber/ro** — the caller is progressively notified as each value is produced, but `Subscribe()` still doesn't return until the pipeline is done:
```go
var pipeline = ro.PipeOp3(
    ro.Filter(func(x int) bool {
        return x%2 == 1
    }),
    ro.Map(func(x int) string {
        return fmt.Sprintf("processed-%d", x)
    }),
    ro.DelayEach[string](100 * time.Millisecond),
)

func main() {
    observable := pipeline(ro.Just(1, 2, 3))

    // Subscribe() blocks here until the pipeline completes (~200ms),
    // but each value is printed as soon as it's ready, not all at once.
    _ = observable.Subscribe(ro.OnNext(func(s string) {
        fmt.Println(s) // "processed-1" at ~100ms, "processed-3" at ~200ms
    }))
}
```

`ro` has two built-in operators for offloading work to its own goroutine: `SubscribeOn` moves the upstream source's execution off the calling goroutine, `ObserveOn` does the same for delivery to your observer — both are meant to decouple producer and consumer speed (backpressure), not to make `Subscribe()` itself return immediately. To do that — run a pipeline without blocking the calling goroutine at all — wrap the subscription in your own `go func() { ... }()`.

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
    numbers := lo.RangeFrom(1, 10) // [1, 2, ..., 10]
    evens := lo.Filter(numbers, func(x int, _ int) bool {
        return x%2 == 0
    })

    // Use ro for real-time processing
    observable := ro.Pipe1(
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

Most rows below aren't "missing" from `lo` — they simply don't apply to synchronous, in-memory collections. `lo` doesn't need backpressure any more than `sort.Slice` does.

:::info Feature Matrix

| Feature               | samber/lo                    | samber/ro                          |
| ---------------------- | ----------------------------- | ----------------------------------- |
| Map/Filter/Reduce      | ✅                             | ✅                                   |
| Error Handling         | Go `error` return values      | Stream-level `Catch`/`Retry`/propagation |
| Concurrency model      | None — plain function calls   | Synchronous by default; opt into goroutines yourself when a source is truly async |
| Time-based Operations  | N/A — no concept of "over time" | ✅ (debounce, throttle, interval)  |
| Backpressure           | N/A — bounded collections     | ✅                                   |
| Hot/Cold Observables   | N/A                            | ✅                                   |
| Subject Types          | N/A                            | ✅                                   |

:::

If a row says N/A for `lo`, that's usually a sign the problem calls for `ro` instead — not that `lo` is behind.

:::tip Learn more

- [Getting started with ro](../getting-started) — install `ro` and see your first pipeline
- [channels vs ro](./channels-vs-ro) — how `ro` relates to native Go concurrency
- [iter vs ro](./iter-vs-ro) — how `ro` relates to Go's `iter` package
- [Observable basics](../core/observable) for reactive concepts
- [Operators guide](../core/operators) for stream transformations
- [Backpressure](../glossary#Backpressure) in reactive systems

:::
