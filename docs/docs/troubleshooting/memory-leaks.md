---
title: 🧠 Memory Leaks
description: Detect, prevent, and fix memory leaks in reactive streams
sidebar_position: 5
---

# 🧠 Memory Leaks

Memory leaks in reactive streams can be particularly insidious because they often accumulate slowly over time. This guide covers how to detect, prevent, and fix memory leaks in `samber/ro` applications.

## 1. Common Memory Leak Patterns

### Unclosed Subscriptions

```go
// ❌ PROBLEM: Subscription never closed
func leakySubscription() {
    source := ro.Interval(1 * time.Second)
    
    // Subscription created but never closed
    subscription := source.Subscribe(ro.OnNext(func(tick int64) {
        fmt.Println("Tick:", tick)
    }))
    
    // source.Subscribe is non-blocking, because ro.Interval returns immediatly
    // Forgot to call subscription.Unsubscribe()
    // This will keep the interval running forever
}
```

**Symptoms:**
- Gradually increasing goroutine count
- Growing memory usage over time
- CPU usage from background goroutines

**Solution:** Always clean up subscriptions:

```go
// ✅ Proper subscription cleanup
func nonLeakySubscription() {
    source := ro.Interval(1 * time.Second)
    subscription := source.Subscribe(ro.OnNext(func(tick int64) {
        fmt.Println("Tick:", tick)
    }))

    // Clean up after use
    defer subscription.Unsubscribe()
    
    // Do work...
    time.Sleep(10 * time.Second)
    // subscription will be automatically unsubscribed
}
```

### Goroutine Leaks from Custom Operators

```go
// ❌ PROBLEM: Goroutine continues after observable is disposed
func leakyOperator() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        go func() {
            ticker := time.NewTicker(100 * time.Millisecond)
            defer ticker.Stop()
            
            for {
                select {
                case <-ticker.C:
                    observer.Next(time.Now().Unix())
                // No way to stop this goroutine!
                }
            }
        }()
        
        return func() {
            // Teardown doesn't stop the goroutine
        }
    })
}
```

**Solution:** Use context for goroutine lifecycle management:

```go
// ✅ Proper goroutine lifecycle management
func nonLeakyOperator() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        ctx, cancel := context.WithCancel(context.Background())
        ticker := time.NewTicker(100 * time.Millisecond)
        
        go func() {
            defer ticker.Stop()
            
            for {
                select {
                case <-ctx.Done():
                    return // Goroutine exits cleanly
                case <-ticker.C:
                    observer.Next(time.Now().Unix())
                }
            }
        }()
        
        return func() {
            cancel() // Signal goroutine to stop
        }
    })
}
```

### Resource Leaks (Files, Network, Database)

```go
// ❌ PROBLEM: Resources not properly cleaned up
func leakyResourceOperator(filename string) ro.Observable[string] {
    return ro.NewObservable(func(observer ro.Observer[string]) ro.Teardown {
        file, err := os.Open(filename)
        if err != nil {
            observer.Error(err)
            return nil
        }
        
        scanner := bufio.NewScanner(file)
        go func() {
            for scanner.Scan() {
                observer.Next(scanner.Text())
            }
            observer.Complete()
            // File never closed!
        }()
        
        return nil
    })
}
```

**Solution:** Always clean up resources in teardown:

```go
// ✅ Proper resource cleanup
func nonLeakyResourceOperator(filename string) ro.Observable[string] {
    return ro.NewObservable(func(observer ro.Observer<string]) ro.Teardown {
        file, err := os.Open(filename)
        if err != nil {
            observer.Error(err)
            return nil
        }
        
        scanner := bufio.NewScanner(file)
        done := make(chan struct{})
        
        go func() {
            defer close(done)
            
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
            file.Close() // Always close the file
        }
    })
}
```

### Infinite Stream Accumulation

```go
// ❌ PROBLEM: Accumulates infinite data
func memoryLeakFromAccumulation() ro.Observable[[]int] {
    return ro.Pipe1(
        ro.Interval(1 * time.Second), // Infinite stream
        ro.Scan(
            func(acc []int, value int64) []int {
                return append(acc, int(value)) // Grows without bound!
            },
            []int{},
        ),
    )
}
```

**Solution:** Use bounded accumulation:

```go
// ✅ Bounded sliding window
func boundedAccumulation(windowSize int) func(ro.Observable[int64]) ro.Observable[[]int] {
    return func(source ro.Observable[int64]) ro.Observable[[]int] {
        return ro.NewObservable(func(observer ro.Observer[[]int]) ro.Teardown {
            window := make([]int, 0, windowSize)
            
            sub := source.Subscribe(ro.NewObserver(
                func(value int64) {
                    window = append(window, int(value))
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

## 2. Detection Tools and Techniques

### Runtime Memory Profiling

```bash
# Generate memory profile
go test -memprofile=mem.prof -bench=.

# Or profile running application
go tool pprof http://localhost:6060/debug/pprof/heap
```

```go
// Programmatic memory profiling
func startMemoryProfiling() {
    f, err := os.Create("mem.prof")
    if err != nil {
        log.Fatal(err)
    }
    
    runtime.GC() // Force GC to get accurate baseline
    pprof.WriteHeapProfile(f)
    f.Close()
}
```

### Custom Memory Monitoring

```go
type MemoryMonitor struct {
    lastGC     time.Time
    lastStats  runtime.MemStats
    mu         sync.Mutex
}

func NewMemoryMonitor() *MemoryMonitor {
    var m runtime.MemStats
    runtime.ReadMemStats(&m)
    
    return &MemoryMonitor{
        lastGC:    time.Now(),
        lastStats: m,
    }
}

func (mm *MemoryMonitor) CheckMemoryUsage() {
    mm.mu.Lock()
    defer mm.mu.Unlock()
    
    var stats runtime.MemStats
    runtime.ReadMemStats(&stats)
    
    // Calculate deltas
    heapAllocDiff := int64(stats.HeapAlloc) - int64(mm.lastStats.HeapAlloc)
    numGCdiff := int64(stats.NumGC) - int64(mm.lastStats.NumGC)
    
    if heapAllocDiff > 10*1024*1024 { // 10MB increase
        log.Printf("🚨 Memory increased by %d bytes", heapAllocDiff)
    }
    
    if numGCdiff > 0 {
        log.Printf("🗑️ GC ran %d times since last check", numGCdiff)
        mm.lastGC = time.Now()
    }
    
    mm.lastStats = stats
}

// Usage in reactive pipeline
func memoryAwareOperator(source ro.Observable[int]) ro.Observable[int] {
    monitor := NewMemoryMonitor()
    
    return ro.Map(func(x int) int {
        result := expensiveOperation(x)
        
        // Check memory usage periodically
        if x%100 == 0 {
            monitor.CheckMemoryUsage()
        }
        
        return result
    })
}
```

### Goroutine Leak Detection

```go
func detectGoroutineLeaks() {
    initialCount := runtime.NumGoroutine()
    
    // Run your reactive code
    runReactivePipeline()
    
    // Force cleanup
    runtime.GC()
    time.Sleep(100 * time.Millisecond)
    
    finalCount := runtime.NumGoroutine()
    leaked := finalCount - initialCount
    
    if leaked > 0 {
        log.Printf("🚨 Detected %d goroutine leaks", leaked)
        
        // Get goroutine stack traces
        buf := make([]byte, 1<<20)
        stackSize := runtime.Stack(buf, true)
        log.Printf("Goroutine stacks:\n%s", buf[:stackSize])
    }
}
```

## 3. Prevention Strategies

### Context-Based Lifecycle Management

```go
// ✅ Always use context for long-running operations
func createContextBasedOperator(ctx context.Context) ro.Observable[int] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
        ticker := time.NewTicker(100 * time.Millisecond)
        defer ticker.Stop()
        
        for {
            select {
            case <-ctx.Done():
                return // Clean exit
            case <-ticker.C:
                observer.Next(time.Now().Unix())
            }
        }
    })
}

// Usage with proper context management
func runWithContext() {
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    observable := createContextBasedOperator(ctx)
    subscription := observable.Subscribe(ro.OnNext(func(value int) {
        fmt.Println("Value:", value)
    }))
    
    // Automatically cancels after 30 seconds
    <-ctx.Done()
    subscription.Unsubscribe()
}
```

### Bounded Buffer Patterns

The `ObserveOn` operator is crucial for controlling synchronization and preventing memory leaks by:

1. **Scheduling emissions on a dedicated goroutine**
2. **Providing backpressure control** through bounded buffering
3. **Preventing goroutine explosion** by limiting concurrent operations

```go
// ✅ Using ObserveOn with bounded buffer to control memory usage
func boundedObserveOnExample() {
    // Create a fast producer that could overwhelm consumers
    fastProducer := ro.Range(1, 1000000)
    
    // Apply ObserveOn with a small buffer to control memory
    boundedStream := ro.Pipe2(
        fastProducer,
        ro.ObserveOn(100), // Buffer only 100 items and send downstream observers into a different goroutine
        ro.Map(func(v int64) int64 {
            time.Sleep(100*time.Millisecond)   // simulate slow processing
            return v
        })
    )
    
    // Process items with controlled memory usage
    subscription := boundedStream.Subscribe(ro.NewObserver[int](
        func(value int) {
            fmt.Printf("Processed: %d\n", value)
        },
        func(err error) {
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            fmt.Println("Stream completed")
        },
    ))
    
    // The buffer will automatically apply backpressure
    // when the consumer can't keep up with the producer
    subscription.Wait()
}
```

## 4. Memory Leak Prevention Checklist

### Development Time
- [ ] All subscriptions have proper cleanup (`defer subscription.Unsubscribe()`)
- [ ] Custom operators use context for goroutine management
- [ ] Resources (files, connections) are closed in teardown functions
- [ ] Infinite streams use bounded buffers or sliding windows

### Code Review
- [ ] Check for goroutine creation without cleanup mechanism
- [ ] Verify all `NewObservable` calls return proper teardown functions
- [ ] Ensure context cancellation is respected in long-running operations
- [ ] Look for accumulation patterns that might grow without bound
- [ ] Verify error paths also clean up resources

### Testing
- [ ] Run tests with race detector (`go test -race`)
- [ ] Test goroutine cleanup after subscription/unsubscription
- [ ] Use go.uber.org/goleak
- [ ] Run long-running integration tests to detect slow leaks
- [ ] Profile memory usage with `pprof`

### Production
- [ ] Set up memory usage monitoring and alerts
- [ ] Track goroutine counts over time
- [ ] Monitor GC frequency and duration
- [ ] Set up heap dump collection on high memory usage
- [ ] Have automated responses to memory alerts

## Next Steps

- [Performance Issues](./performance) - Performance optimization techniques
- [Concurrency Issues](./concurrency) - Race conditions and goroutine management
- [Common Issues](./common-issues) - Frequently encountered problems
