---
title: ✌️ About
description: Discover "ro", the Reactive Programming for Go
sidebar_position: 0
---

# ✌️ About

**ro** is a reactive programming library for Go that brings the power of [`Observable`](./core/observable) streams to the Go ecosystem. Inspired by ReactiveX patterns, `ro` enables developers to work with data streams using a declarative, composable API.

`samber/ro` is like `samber/lo`, but for events.

Reactive programming treats events as streams that can be observed, transformed, and composed. This paradigm shift makes complex event-driven systems more manageable and maintainable.

## What is Reactive Programming?

:::tip Paradigm Shift

Reactive programming is a programming paradigm focused on event-driven applications and the propagation of change. It allows you to:

- **Handle asynchronous events** naturally and consistently
- **Transform and compose** data streams declaratively
- **Manage backpressure** and resource usage efficiently
- **Build responsive** and resilient applications

:::

## Why Use ro?

### 1. **Simplified event-driven logic**

Replace complex callback chains with clean, declarative stream operations:

```go
// Instead of nested callbacks
observable := ro.Pipe[int, string](
    ro.Just(0, 1, 2, 3, 4, 5),
    ro.Filter(func(x int) bool {
        return x%2 == 0
    }),
    ro.Map(func(x int) string {
        return fmt.Sprintf("even-%d", x)
    }),
)

subscription := observable.Subscribe(
    ro.NewObserver(
        func(v string) { ... },  // on value
        func(err error) { ... }, // on error
        func() { ... },          // on completion
    ),
)
```

### 2. **Powerful Operators**

`ro` provides a rich set of [operators](./core/operators) for stream manipulation:

```go
// Combine multiple streams
combined := ro.Merge(stream1, stream2)

// Handle errors gracefully
observable := ro.Pipe[string, string](
    combined,
    ro.Catch(func(err error) ro.Observable[string] {
        return ro.Just("fallback-value")
    }),
    ro.DelayEach(100 * time.Millisecond),
)

subscription := observable.Subscribe(
    ro.NewObserver(
        func(v string) { ... },  // on value
        func(err error) { ... }, // on error
        func() { ... },          // on completion
    ),
)
```

### 3. **Resource Management**

:::warning Automatic Cleanup

Automatic cleanup and backpressure handling prevent resource leaks. See [Subscription](./core/subscription) for proper resource management patterns.

:::

```go
// Automatically cancel when stream is completed
observable := ro.Pipe[int64, int64](
    ro.Interval(1 * time.Second),
    ro.Take(10),
)

subscription := observable.Subscribe(
    ro.NewObserver(
        func(v int64) { ... },  // on value
        func(err error) { ... }, // on error
        func() { ... },          // on completion
    ),
)
```

## Design Principles

### **Go-idiomatic API**

While inspired by ReactiveX and `rxjs`, `ro` embraces Go's conventions:
- Context-aware operations
- Error handling via multiple return values
- Goroutine-safe by design
- Zero allocations and limited lock in hot paths where possible

### **Type Safety**

:::tip Compile-time Safety

Strong typing prevents runtime errors and enables better tooling support:

:::
```go
// Compile-time type checking
obs := ro.Just(1, 2, 3)             // Observable[int]
subscription := ro.Map(mapper)(obs) // mapper must be func(int) T
```

`ro.Pipe` receives `any` parameters but multiple type-safe variants are available:

```go
obs := ro.Pipe3(
    ro.Range(0, 42),
    ro.Filter(func(x int64) bool {
        return x%2 == 0
    }),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("even-%d", x)
    }),
    ro.Take[string](10),
)
```

### **Performance Focus**

:::danger Performance-First Design

Designed for high-throughput scenarios:
- Minimal allocations
- Efficient backpressure propagation
- Operator fusion opportunities
- Zero runtime reflection
- Limited locks

:::

## Why this name?

I wanted to name it `$o`, but I think Go is not ready for special characters in package name 😁. `ro` is a *short name*, similar to `rx` and no Go package uses this name.

## When to Use ro?

`ro` excels in scenarios involving:
- **Real-time data processing** (WebSocket events, sensor data)
- **User interface events** (clicks, keystrokes, form inputs)
- **API response handling** (with retry, timeout, and caching)
- **Data processing** with transformation, aggregation and enrichment
- **Event-driven** patterns

See [comparisons](./comparison/lo-vs-ro) with other Go libraries for more context.

`ro` brings the elegance and power of reactive programming to Go while maintaining the language's core strengths of simplicity and performance.
