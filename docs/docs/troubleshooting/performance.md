---
title: Performance Issues
description: Identify and resolve performance bottlenecks
sidebar_position: 4
---

# ⚡ Performance Issues

Performance problems in reactive streams can be subtle and difficult to diagnose. This guide covers common performance issues and how to resolve them in `samber/ro` applications.

## 1. Backpressure Problems

### Fast Producer, Slow Consumer

```go
// ❌ PROBLEM: Producer overwhelms consumer
func fastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000000; i++ {
            observer.Next(i) // Produces as fast as possible
        }
        observer.Complete()
        return nil
    })
}

func slowConsumer() {
    fastProducer().Subscribe(ro.OnNext(func(x int) {
        time.Sleep(1 * time.Millisecond) // Slow processing
        fmt.Println(x)
    }))
    // Result: Memory usage explodes, goroutine blocking
}
```

**Solutions:**

#### Option 1: Buffer with Overflow Strategy
```go
// ✅ Buffered with overflow handling
func fastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000000; i++ {
            observer.Next(i) // Produces as fast as possible
        }
        observer.Complete()
        return nil
    })
}

func slowConsumer() {
    obs := ro.Pipe2(
        fastProducer(),
        ro.ObserveOn(100), // buffer of size=100
    )
    obs.Subscribe(ro.OnNext(func(x int) {
        time.Sleep(1 * time.Millisecond) // Slow processing
        fmt.Println(x)
    }))
}
```

#### Option 2: Throttle Production
```go
// ✅ Rate-limited consumer
func fastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000000; i++ {
            observer.Next(i) // Produces as fast as possible
        }
        observer.Complete()
        return nil
    })
}

func slowConsumer() {
    obs := ro.Pipe2(
        fastProducer(),
        ro.ThrottleTime(10*time.Millisecond), // at most 100 values per second
    )
    obs.Subscribe(ro.OnNext(func(x int) {
        time.Sleep(1 * time.Millisecond) // Slow processing
        fmt.Println(x)
    }))
}
```

#### Option 3: Use Built-in Backpressure
```go
// ✅ Combine with delay for natural backpressure
func backpressureAware() ro.Observable[int] {
    return ro.Pipe2(
        ro.Just(generateLargeDataset()),
        ro.DelayEach(1 * time.Millisecond), // Adds natural backpressure
    )
}
```

## 2. Inefficient Operator Patterns

### Excessive Allocations

```go
// ❌ PROBLEM: Creating many temporary objects
func memoryIntensiveOperator(source ro.Observable[string]) ro.Observable[string] {
    return ro.Map(func(s string) string {
        // Creates new string and slice for every value
        words := strings.Fields(s)
        result := make([]string, 0, len(words))
        for _, word := range words {
            result = append(result, strings.ToUpper(word))
        }
        return strings.Join(result, " ")
    })
}
```

**Solution:** Reduce allocations with object pooling:

```go
// ✅ Memory-efficient with pooling
var stringBuilderPool = sync.Pool{
    New: func() interface{} {
        return &strings.Builder{}
    },
}

func memoryEfficientOperator(source ro.Observable[string]) ro.Observable<string] {
    return ro.Map(func(s string) string {
        builder := stringBuilderPool.Get().(*strings.Builder)
        defer func() {
            builder.Reset()
            stringBuilderPool.Put(builder)
        }()
        
        // Process using reusable builder
        scanner := bufio.NewScanner(strings.NewReader(s))
        first := true
        for scanner.Scan() {
            if !first {
                builder.WriteString(" ")
            }
            first = false
            builder.WriteString(strings.ToUpper(scanner.Text()))
        }
        
        return builder.String()
    })
}
```

## 3. Memory Usage Optimization

### Large Intermediate Collections

```go
// ❌ PROBLEM: Keeps all intermediate values in memory
func memoryHeavyProcessing(source ro.Observable[LargeObject]) ro.Observable[ProcessedObject] {
    return ro.Pipe2(
        source,
        ro.Map(func(obj LargeObject) LargeObject {
            return preprocess(obj) // Creates many intermediate objects
        }),
        ro.Map(func(obj LargeObject) ProcessedObject {
            return process(obj) // More intermediate objects
        }),
    )
}
```

**Solution:** Stream processing without retention:

```go
// ✅ Process immediately, don't retain
func memoryEfficientProcessing(source ro.Observable[LargeObject]) ro.Observable[ProcessedObject] {
    return ro.Map(func(obj LargeObject) ProcessedObject {
        // Process and discard intermediate objects immediately
        preprocessed := preprocess(obj)
        result := process(preprocessed)
        // preprocessed is eligible for GC here
        return result
    })
}
```

### Infinite Stream Accumulation

```go
// ❌ PROBLEM: Accumulates infinite data
func accumulatingStream() ro.Observable[[]int] {
    return ro.Scan(
        func(acc []int, value int) []int {
            return append(acc, value) // Grows without bound!
        },
        []int{},
    )
}

// With infinite source like ro.Interval, this will eventually OOM
```

**Solution:** Bounded accumulation:

```go
// ✅ Bounded window accumulation
func slidingWindow(windowSize int) func(ro.Observable[int]) ro.Observable[[]int] {
    return func(source ro.Observable[int]) ro.Observable[[]int] {
        return ro.NewObservable(func(observer ro.Observer[[]int]) ro.Teardown {
            window := make([]int, 0, windowSize)
            
            sub := source.Subscribe(ro.NewObserver(
                func(value int) {
                    window = append(window, value)
                    if len(window) > windowSize {
                        // Remove oldest element
                        window = window[1:]
                    }
                    
                    // Send copy to prevent external mutation
                    windowCopy := make([]int, len(window))
                    copy(windowCopy, window)
                    observer.Next(windowCopy)
                },
                observer.Error,
                observer.Complete,
            ))
            return sub.Unsubscribe
        })
    }
}
```

## 4. CPU Performance Optimization

### Inefficient Transformations

```go
// ❌ PROBLEM: Repeated expensive computations
func expensiveTransform(source ro.Observable[string]) ro.Observable[string] {
    return ro.Map(func(s string) string {
        // This regex compilation is expensive and repeated
        regex := regexp.MustCompile(`[a-zA-Z]+`)
        matches := regex.FindAllString(s, -1)
        
        result := make([]string, 0, len(matches))
        for _, match := range matches {
            result = append(result, strings.Title(match))
        }
        return strings.Join(result, " ")
    })
}
```

**Solution:** Pre-compile and cache:

```go
// ✅ Pre-compile regex and reuse
var regex = regexp.MustCompile(`[a-zA-Z]+`)

func efficientTransform(source ro.Observable[string]) ro.Observable[string] {
    return ro.Map(func(s string) string {
        matches := regex.FindAllString(s, -1)
        
        // Pre-allocate slice with known capacity
        result := make([]string, 0, len(matches))
        for _, match := range matches {
            result = append(result, strings.Title(match))
        }
        return strings.Join(result, " ")
    })
}
```

## 5. Performance Monitoring and Benchmarking

### Benchmarking Operators

```go
func BenchmarkMapOperator(b *testing.B) {
    source := ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    operator := ro.Map(func(x int) int { return x * 2 })

    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        values, err := ro.Collect(operator(source))
        if err != nil {
            b.Fatal(err)
        }
        _ = values
    }
}

func BenchmarkConcurrentProcessing(b *testing.B) {
    source := ro.Just(make([]int, 1000)...) 
    
    b.Run("Serial", func(b *testing.B) {
        operator := serialProcessing()
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            ro.Collect(operator(source))
        }
    })
    
    b.Run("Parallel", func(b *testing.B) {
        operator := parallelProcessing(10)
        b.ResetTimer()
        for i := 0; i < b.N; i++ {
            ro.Collect(operator(source))
        }
    })
}
```

### Subscriber Concurrency Modes

High-throughput sources can avoid unnecessary synchronization by selecting the right subscriber implementation. The core library now exposes `NewSingleProducerSubscriber`/`NewSingleProducerObservableWithContext`, and operators such as `Range` automatically opt into the `ConcurrencyModeSingleProducer` fast-path when there is exactly one upstream writer. This mode bypasses the `Lock`/`Unlock` calls entirely while retaining panic safety and teardown behavior. Use the following guidance when choosing a mode:

| Concurrency mode | Locking strategy | Drop policy | Recommended usage |
| --- | --- | --- | --- |
| `ConcurrencyModeSafe` | `sync.Mutex` | Blocks producers | Multiple writers or callbacks that may concurrently re-enter observers |
| `ConcurrencyModeEventuallySafe` | `sync.Mutex` | Drops when contended | Fan-in scenarios where losing values is acceptable |
| `ConcurrencyModeUnsafe` | No-op lock wrapper | Blocks producers | Single writer, but still routes through the locking API surface |
| `ConcurrencyModeSingleProducer` | No locking | Blocks producers | Single writer that needs the lowest possible overhead |

Note on panic-capture interaction
: The library still provides a global panic-capture toggle (`ro.SetCaptureObserverPanics`), but for benchmarks and experiments prefer opting out per-subscription. Disabling capture lets some fast-paths (for example the single-producer and unsafe modes) avoid wrapping observer callbacks in the usual defer/recover machinery, which reduces per-notification overhead. Use `ro.WithObserverPanicCaptureDisabled(ctx)` when subscribing in benchmarks to avoid mutating global state and to keep tests parallel-friendly.

Run the million-row benchmark to compare the trade-offs:

```bash
go test -run=^$ -bench BenchmarkMillionRowChallenge -benchmem ./testing
```

Running the benchmark (tips)
:
- The benchmark harness in `testing/benchmark_million_rows_test.go` disables panic capture for the duration of the bench using a per-subscription context opt-out so the harness doesn't mutate global state. If you want to reproduce realistic production numbers, run the benchmark both with capture enabled and disabled.
- Increase bench time to reduce noise:

```bash
go test -run=^$ -bench BenchmarkMillionRowChallenge -benchmem ./testing -benchtime=10s
```

- To check for races, run:

```bash
go test -race ./...
```

- To profile CPU or mutex contention, use `pprof` with the benchmark or a traced run and inspect lock profiles to see how much time is spent acquiring `sync.Mutex` vs useful work.


Sample results on a 1M element pipeline:

- `single-producer`: 60.3 ms/op, 1.5 KiB/op, 39 allocs/op【9dc40c†L1-L5】【f63774†L1-L2】
- `unsafe-mutex`: 63.2 ms/op, 1.5 KiB/op, 39 allocs/op【f63774†L1-L2】【604fb9†L1-L2】
- `safe-mutex`: 67.1 ms/op, 1.6 KiB/op, 40 allocs/op【604fb9†L1-L2】【9ecf78†L1-L4】

The single-producer path trims roughly 4–6% off the runtime compared to the previous `unsafe` mode while preserving allocation characteristics. Stick with the safe variants whenever multiple goroutines might call `Next` concurrently.

## 6. Performance Optimization Checklist

### Memory Optimization
- [ ] Avoid unnecessary allocations in hot paths
- [ ] Use object pools for frequently created objects
- [ ] Pre-allocate slices and maps with known capacity
- [ ] Ensure proper cleanup of goroutines and resources
- [ ] Use bounded buffers for infinite streams

### CPU Optimization
- [ ] Cache expensive computations
- [ ] Avoid O(n²) algorithms in stream processing
- [ ] Profile CPU usage with `pprof`

### Concurrency Optimization
- [ ] Limit concurrent goroutines with semaphores
- [ ] Use appropriate worker pool sizes
- [ ] Implement proper backpressure mechanisms
- [ ] Check for race conditions with `-race` flag
- [ ] Ensure context cancellation is respected

### Monitoring
- [ ] Add performance metrics collection
- [ ] Set up memory and CPU profiling
- [ ] Monitor goroutine counts in production
- [ ] Track error rates and latencies
- [ ] Set up alerts for performance degradation

## Next Steps

- [Memory Leaks](./memory-leaks) - Memory leak detection and prevention
- [Concurrency Issues](./concurrency) - Race conditions and synchronization
- [Debugging Techniques](./debugging) - Systematic debugging approaches