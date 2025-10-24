---
title: Scheduling
description: Learn about scheduling operators in samber/ro and how they differ from other reactive libraries.
sidebar_position: 7
---

# ⚡ Scheduling

Scheduling in reactive programming typically involves controlling which thread or goroutine executes different parts of a reactive pipeline. In `samber/ro`, scheduling is simplified and leverages Go's first-class builtin concurrency.

## Scheduling Operators in samber/ro

Despite Go's excellent concurrency model, `samber/ro` provides two scheduling operators for specific use cases:

### SubscribeOn: Move Upstream to Goroutine

`SubscribeOn` moves the upstream subscription and emissions to a separate goroutine, allowing downstream to start immediately.

```go
// SubscribeOn moves upstream processing to a goroutine
pipeline := ro.Pipe2(
    ro.Just(1, 2, 3, 4, 5),
    ro.SubscribeOn[int](10), // Buffer of 10 items
    ro.Map(func(value int) int {
        fmt.Println("Upstream processing:", value) // Runs in goroutine
        return value * 2
    }),
)

subscription := pipeline.Subscribe(ro.OnNext(func(value int) {
    fmt.Println("Downstream received:", value) // Runs in main goroutine
}))

// Output: Upstream processing starts immediately
// Downstream processes as items become available
```

**When to use SubscribeOn**:
- **Slow upstream operations**: Network calls, database queries, file I/O
- **Blocking operations**: Prevents slow source from blocking subscription
- **Parallel processing**: Start upstream while setting up downstream

```go
// Good: Use SubscribeOn for slow data sources
pipeline := ro.Pipe2(
    fetchFromDatabase(), // Slow database operation
    ro.SubscribeOn[Record](50), // Buffer and run in goroutine
    ro.Map(processRecord), // Fast downstream processing
)
```

### ObserveOn: Move Downstream to Goroutine

`ObserveOn` moves downstream observer callbacks to a separate goroutine, allowing upstream emissions to continue immediately.

```go
// ObserveOn moves downstream processing to a goroutine
pipeline := ro.Pipe2(
    ro.Just(1, 2, 3, 4, 5),
    ro.Map(func(value int) int {
        fmt.Println("Upstream processing:", value) // Runs in main goroutine
        return value * 2
    }),
    ro.ObserveOn[int](10), // Buffer of 10 items
)

subscription := pipeline.Subscribe(ro.OnNext(func(value int) {
    fmt.Println("Downstream received:", value) // Runs in goroutine
}))

// Output: Upstream processing completes immediately
// Downstream processes in background goroutine
```

**When to use ObserveOn**:
- **Slow downstream operations**: UI updates, database writes, API calls
- **Non-blocking observers**: Prevent slow observers from blocking source
- **Background processing**: Handle results while source continues emitting

```go
// Good: Use ObserveOn for slow observers
pipeline := ro.Pipe2(
    ro.Interval(10 * time.Millisecond), // Fast source
    ro.Map(prepareData), // Fast transformation
    ro.ObserveOn[DataItem](100), // Buffer slow observer processing
)

subscription := pipeline.Subscribe(ro.OnNext(func(data DataItem) {
    writeToDatabase(data) // Slow operation in goroutine
}))
```

## Buffer Configuration

Both operators use buffered channels to manage backpressure:

```go
ro.SubscribeOn[int](bufferSize)  // Buffer size controls backpressure
ro.ObserveOn[int](bufferSize)     // Buffer size controls backpressure
```

- **Small buffer (1-100)**: Tight backpressure, upstream blocks frequently
- **Medium buffer (100-1_000)**: Balanced, good for most use cases
- **Large buffer (1_000+)**: Loose backpressure, allows more buffering

## Summary

`samber/ro` scheduling operators provide a bridge between reactive programming patterns and Go's native concurrency model. While you can often achieve the same results with plain goroutines and channels, the scheduling operators offer:

- **Operator-based syntax**: Consistent with other reactive patterns
- **Built-in backpressure**: Channel-based flow control
- **Context propagation**: Automatic context handling
- **Buffer management**: Configurable buffering strategies

In Go, you often don't need the complex scheduling mechanisms required in other languages, but when you do want to combine reactive operators with goroutine-based concurrency, `ObserveOn` and `SubscribeOn` provide clean, idiomatic solutions.

---

**Related Topics:**
- [Backpressure](./backpressure) - Understanding flow control
- [Observable](./observable) - Understanding data producers
- [Operators](../operator) - Data transformation operations