---
title: Operators
description: Learn about operators - the building blocks for transforming Observable streams in samber/ro
sidebar_position: 3
---

# ⚙️ Operators

**Operators** are the building blocks of reactive programming in `samber/ro`. They are functions that transform, filter, combine, or manipulate [`Observable`](./observable) streams, enabling you to build complex data processing pipelines declaratively.

Operators are called sequentially for every messages entering the pipeline. By default, no concurrency is allowed.

## What are Operators?

`Operators` are:
- **Pure functions**: They take an `Observable[A]` and return a new `Observable[B]`
- **Transformers**: They modify the stream of values without changing the source
- **Composable**: They can be chained together to build complex pipelines
- **Lazy**: They only execute when subscribed to

## Using Operators

Many types of operators are available in the library:
- **Creation operators**: The data source, usually the first argument of `ro.Pipe`
- **Chainable operators**: They filter, validate, transform, enrich... messages
  - **Transforming operators**: They transform items emitted by an `Observable`
  - **Filtering operators**: They selectively emit items from a source `Observable`
  - **Conditional operators**: Boolean operators
  - **Math and aggregation operators**: They perform basic math operations
  - **Error handling operators**: They help to recover from error notifications from an `Observable`
  - **Combining operators**: Combine multiple `Observable` into one
  - **Connectable operators**: Convert cold into hot `Observable`
  - **Other**: manipulation of context, utility, async scheduling...
- **Plugins**: External operators (mostly IOs and library wrappers)

See the [operator reference](../operator/creation) for detailed documentation of specific operators.

### 1. Pipe Function (Recommended)

:::tip Pipe Function

The Pipe function provides cleaner syntax and better type safety. The number suffix indicates how many operators you're chaining. Always prefer Pipe over method chaining for better readability and type safety.

:::

```go
// Use Pipe for clean, readable composition
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.Filter(func(x int) bool {
        return x%2 == 0
    }),
    ro.Map(func(x int) string {
        return fmt.Sprintf("even-%d", x)
    }),
)

obs.Subscribe(ro.OnNext(func(s string) {
    fmt.Println(s) // "even-2", "even-4", "even-6", "even-8", "even-10"
}))
```

For stronger type-safety, use `ro.PipeX` variants:

```go
// Use Pipe3 for compile-time type checks
obs := ro.Pipe3(
    source,
    operator1,
    operator2,
    operator3,
)
```

`ro.PipeX` variants can be used as an operator:

```go
// Use PipeOp for pipeline composition
obs := ro.Pipe3(
    source,
    operator1,
    // sub-pipeline
    ro.PipeOp2(
        operator3,
        operator4,
    ),
    operator5,
)
```

### 2. Method Chaining

Method chaining works but can become hard to read. Each operator takes the source and returns a new observable.

```go
// Chain operators directly
source := ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),

obs1 := ro.Filter(func(x int) bool {
    return x%2 == 0 
})(source)
obs2 := ro.Map(func(x int) string {
    return fmt.Sprintf("even-%d", x)
})(obs1)

obs2.Subscribe(ro.OnNext(func(s string) {
    fmt.Println(s)
}))
```

## Operator Pipelines

### Complex Data Processing

Build sophisticated data pipelines by composing multiple operators. This example filters events, extracts data, and removes duplicates.

```go
// Real-world example: process user events
source := ro.Just(
    UserEvent{ID: 1, Action: "click", Timestamp: time.Now()},
    UserEvent{ID: 2, Action: "scroll", Timestamp: time.Now()},
    UserEvent{ID: 3, Action: "click", Timestamp: time.Now()},
)

obs := ro.Pipe4(
    source,
    ro.Filter(func(event UserEvent) bool {
        return event.Action == "click" // Keep only clicks
    }),
    ro.Map(func(event UserEvent) int {
        return event.ID // Extract user IDs
    }),
    ro.Distinct(), // Remove duplicate user IDs
    ro.Take(10), // Limit to 10 users
)

obs.Subscribe(ro.OnNext(func(userID int) {
    fmt.Println("Active user:", userID)
}))
```

### Error Recovery Pipeline

:::warning Error Handling

Create resilient pipelines that handle failures gracefully. This pattern uses fallible operators, retry logic, and fallback strategies. See [Error Handling](../troubleshooting/wip) for more details.

:::

```go
// Robust pipeline with error handling
var pipeline = ro.PipeOp4(
    ro.MapErr(func(item string) (string, error) {
        // Simulate processing that might fail
        if len(item) < 3 {
            return "", fmt.Errorf("item too short: %s", item)
        }
        return strings.ToUpper(item), nil
    }),
    ro.RetryWithConfig(RetryConfig{
        MaxRetries: 3,
        Delay:      100 * time.Millisecond,
    }),
    ro.Catch(func(err error) ro.Observable[string] {
        fmt.Println("Retries exhausted, using fallback")
        return ro.Just("FALLBACK")
    }),
    ro.Timeout(5 * time.Second),
)

obs := pipeline(ro.Just("hi", "hello", "world", "a"))
obs.Subscribe(ro.NewObserver(
    func(value string) { fmt.Println("Result:", value) },
    func(err error) { fmt.Println("Pipeline error:", err) },
    func() { fmt.Println("Pipeline completed") },
))
```

## Custom Operators

Create **reusable operators** to encapsulate common transformations and share them across your application.

```go
// Custom operator that squares numbers
func Square[T constraints.Integer]() func (ro.Observable[T]) ro.Observable[T] {
    return func(source Observable[T]) Observable[T] {
        return NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination Observer[T]) Teardown {
            sub := source.SubscribeWithContext(
                subscriberCtx,
                NewObserverWithContext(
                    func(ctx context.Context, value T) {
                        destination.NextWithContext(ctx, value*value)
                    },
                    destination.ErrorWithContext,
                    destination.CompleteWithContext,
                ),
            )

            return sub.Unsubscribe
        })
    }
}

// Usage
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    Square[int](),
)
obs.Subscribe(ro.OnNext(func(x int) {
    fmt.Println(x) // 1, 4, 9, 16, 25
}))
```

More info on custom operators in the [🏴‍☠️ hacking](../hacking.md) section.

Many more operators are available in the [source code](https://github.com/samber/ro) of the project.

## Operator Best Practices

### 1. Prefer Pipe over Method Chaining

:::tip Best Practice

Always use the Pipe function for cleaner syntax and better type safety. Method chaining can become hard to read and maintain.

:::

```go
// ✅ Good: Clean Pipe syntax
obs := ro.Pipe3(
    source,
    ro.Filter(predicate),
    ro.Map(transformer),
    ro.Take[int](10),
)

// ⚠️ Valid, but bad syntax: Method chaining
obs := Take[int](10, Map(transformer, Filter(predicate, source)))
```

### 2. Use Type-safe Pipe Variants

Use typed Pipe variants (Pipe2, Pipe3, etc.) for compile-time type checking. The generic Pipe function is more flexible but provides less type safety.

```go
// ✅ Good: Compile-time type checking
obs := ro.Pipe2(source, filterOp, mapOp)

// ⚠️ Works but less type safety
obs := ro.Pipe[int, int](source, filterOp, mapOp)
```

### 3. Handle Memory Leaks

:::warning Memory Management

Bound infinite streams with operators like `Take`, `TakeUntil`, or `TakeWhile` to prevent memory leaks. Unbounded streams can quickly exhaust system resources.

:::

```go
// ✅ Good: Limit infinite streams with the `ro.TakeUntil` operator
bounded := ro.Pipe1(
    ro.Interval(1 * time.Second),
    ro.TakeUntil[int64](ro.Timer(30 * time.Second)),
)

// ⚠️ Risky: Unbounded stream
unbounded := ro.Interval(1 * time.Second) // May leak memory
```

### 4. Consider Backpressure

:::warning Backpressure

Handle fast producers with appropriate backpressure mechanisms. In `ro`, backpressure is handled naturally through blocking behavior, but you may need additional buffering for extreme cases.

See [Observer vs Go Channels](./observer#Observer-vs-Go-Channels) for more details on backpressure.

:::

```go
// ✅ Good: Handle fast producers
obs := ro.Pipe2(
    fastProducer,        // Emits values rapidly
    ro.Buffer(1000),     // Buffer some values
    ro.DelayEach(10 * time.Millisecond), // Slow down consumption
)

// ⚠️ Problematic: No backpressure handling
fast.Subscribe(slowObserver) // May overwhelm the observer
```

Operators are the core building blocks that make reactive programming powerful and expressive. By composing operators, you can create complex data processing pipelines that are both declarative and efficient.