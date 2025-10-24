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