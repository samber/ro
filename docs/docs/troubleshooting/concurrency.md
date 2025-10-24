---
title: 🔄 Concurrency Issues
description: Understanding and resolving race conditions, deadlocks, and synchronization problems
sidebar_position: 6
---

# 🔄 Concurrency Issues

Concurrency problems in reactive streams can be particularly tricky because they manifest intermittently and are hard to reproduce. This guide covers common concurrency issues and their solutions in `samber/ro` applications.

## 1. Safe vs Unsafe Observables

### Understanding the Difference

`ro.NewSafeObservable` and `ro.NewUnsafeObservable` have different concurrency guarantees:

- **Unsafe Observable**: Higher performance, but no protection against concurrent calls to `destination.Next()`
- **Safe Observable**: Slight performance overhead, but protects against concurrent access

### Race Condition with Unsafe Observable

```go
// ❌ PROBLEM: Concurrent access with unsafe observable
func raceConditionExample() {
    observable := ro.NewUnsafeObservable(func(observer ro.Observer[int]) ro.Teardown {
        var wg sync.WaitGroup

        // Simulate concurrent producers
        for i := 0; i < 3; i++ {
            wg.Add(1)
            go func(id int) {
                for j := 0; j < 10; j++ {
                    observer.Next(id*10 + j) // Race condition here!
                }
            }(i)
        }
        
        return func() {
            wg.Wait()
        }
    })
    
    observable.Subscribe(ro.OnNext(func(value int) {
        fmt.Println("Received:", value)
    }))
    // Results are unpredictable, values may be lost or corrupted
}
```

**Solution:** Use Safe Observable for concurrent producers:

```go
// ✅ Safe concurrent access
func safeConcurrentExample() {
    observable := ro.NewSafeObservable(func(observer ro.Observer[int]) ro.Teardown {
        var wg sync.WaitGroup
        
        for i := 0; i < 3; i++ {
            wg.Add(1)
            go func(id int) {
                defer wg.Done()
                for j := 0; j < 10; j++ {
                    observer.Next(id*10 + j) // Safe concurrent access
                }
            }(i)
        }
        
        return func() {
            wg.Wait()
        }
    })
    
    observable.Subscribe(ro.OnNext(func(value int) {
        fmt.Println("Received:", value)
    }))
    // All values are received reliably
}
```

### When to Use Each Observable Type

```go
// ✅ Use Unsafe when producer is single-threaded
func singleThreadedProducer() ro.Observable[int] {
    return ro.NewUnsafeObservable(func(observer ro.Observer[int]) ro.Teardown {
        // Single goroutine producing values
        for i := 0; i < 100; i++ {
            observer.Next(i)
        }
        observer.Complete()
        return nil
    })
}

// ✅ Use Safe when multiple goroutines call observer.Next()
func multiThreadedProducer() ro.Observable[int] {
    return ro.NewSafeObservable(func(observer ro.Observer[int]) ro.Teardown {
        var wg sync.WaitGroup
        
        // Multiple goroutines producing values
        for i := 0; i < 5; i++ {
            wg.Add(1)
            go func(id int) {
                defer wg.Done()
                for j := 0; j < 20; j++ {
                    observer.Next(id*20 + j)
                }
            }(i)
        }
        
        return func() {
            wg.Wait()
            observer.Complete()
        }
    })
}
```

## 2. Context Cancellation Issues

### Context Propagation Problems

```go
// ❌ PROBLEM: Context not properly propagated
func brokenContextPropagation() ro.Observable[string] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[string]) ro.Teardown {
        // Context is received but not used
        go func() {
            for i := 0; i < 1000; i++ {
                time.Sleep(10 * time.Millisecond)
                observer.Next(fmt.Sprintf("item-%d", i))
                // Ignores context cancellation!
            }
            observer.Complete()
        }()
        
        return func() {
            // Teardown doesn't stop the goroutine
        }
    })
}
```

**Solution:** Proper context usage:

```go
// ✅ Proper context propagation and cancellation
func properContextPropagation() ro.Observable[string] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[string]) ro.Teardown {
        done := make(chan struct{})
        
        go func() {
            defer close(done)
            
            for i := 0; i < 1000; i++ {
                select {
                case <-ctx.Done():
                    return // Respect context cancellation
                default:
                    time.Sleep(10 * time.Millisecond)
                    observer.Next(fmt.Sprintf("item-%d", i))
                }
            }
            
            observer.Complete()
        }()
        
        return func() {
            // Signal goroutine to stop
            <-done // Wait for goroutine to finish
        }
    })
}
```

### Context Deadlock

```go
// ❌ PROBLEM: Context misuse causing deadlock
func contextDeadlock() ro.Observable[int] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
        // This will deadlock if parent context is already cancelled
        childCtx, cancel := context.WithCancel(ctx)
        
        go func() {
            // If parent is cancelled, this blocks forever
            <-childCtx.Done()
            observer.Error(childCtx.Err())
        }()
        
        return cancel
    })
}
```

**Solution:** Check context before creating derived contexts:

```go
// ✅ Safe context usage
func safeContextUsage() ro.Observable[int] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
        // Check if parent is already cancelled
        if ctx.Err() != nil {
            observer.Error(ctx.Err())
            return nil
        }
        
        childCtx, cancel := context.WithCancel(ctx)
        
        go func() {
            select {
            case <-childCtx.Done():
                observer.Error(childCtx.Err())
            case <-time.After(5 * time.Second):
                observer.Complete()
            }
        }()
        
        return cancel
    })
}
```

## 3. Synchronization Issues

### Shared State Race Conditions

```go
// ❌ PROBLEM: Shared state without synchronization
func Count[T any]() func(ro.Observable[T]) ro.Observable[T] {
    // Every subscriptions will share the same counter.
    var counter int
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewObservable(func(observer ro.Observer[T]) ro.Teardown {
            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    counter++
                    observer.Next(value)
                },
                observer.Error,
                observer.Complete,
            ))
            return sub.Unsubscribe
        })
    }
}
```

**Solution:** No side-effect:

```go
// ✅ Synchronized shared state
func Count[T any]() func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewObservable(func(observer ro.Observer[T]) ro.Teardown {
            // Every subscriptions gets is own counter.
            var counter int

            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    counter++
                    observer.Next(value)
                },
                observer.Error,
                observer.Complete,
            ))
            return sub.Unsubscribe
        })
    }
}
```

## 4. Deadlock Scenarios

### Context Deadlock in Pipeline

```go
// ❌ PROBLEM: Context misuse causing deadlock
func contextPipelineDeadlock() ro.Observable[string] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer<string]) ro.Teardown {
        // Creating derived context in goroutine can cause issues
        go func() {
            childCtx, cancel := context.WithCancel(ctx)
            defer cancel()
            
            select {
            case <-childCtx.Done():
                // This can block if parent and child are both cancelled
                observer.Error(childCtx.Err())
            case <-time.After(1 * time.Second):
                observer.Next("timeout")
            }
        }()
        
        return func() {}
    })
}
```

**Solution:** Avoid context operations in goroutines without proper synchronization:

```go
// ✅ Safe context usage in pipeline
func safeContextPipeline() ro.Observable[string] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer<string]) ro.Teardown {
        // Create derived context in main goroutine
        childCtx, cancel := context.WithCancel(ctx)
        
        go func() {
            defer cancel()
            
            select {
            case <-childCtx.Done():
                if ctx.Err() != nil {
                    observer.Error(ctx.Err())
                }
                return
            case <-time.After(1 * time.Second):
                observer.Next("timeout")
                observer.Complete()
            }
        }()
        
        return cancel
    })
}
```

## 5. Concurrency Debugging Tools

### Goroutine State Inspection

```go
func PrintGoroutineStacks() {
    buf := make([]byte, 1<<20)
    stackSize := runtime.Stack(buf, true)
    fmt.Printf("Goroutine stacks:\n%s", buf[:stackSize])
}

func MonitorGoroutines() {
    ticker := time.NewTicker(10 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        count := runtime.NumGoroutine()
        fmt.Printf("Current goroutine count: %d\n", count)
        
        if count > 100 { // Arbitrary threshold
            fmt.Printf("High goroutine count detected!\n")
            PrintGoroutineStacks()
        }
    }
}
```

### Race Condition Detection

```bash
# Run tests with race detection
go test -race ./...

# Run application with race detection
go run -race main.go

# Build with race detection
go build -race -o myapp
./myapp
```

### Concurrency Testing Helper

```go
func RunConcurrentTest(
    t *testing.T,
    numGoroutines int,
    iterationsPerGoroutine int,
    testFunc func(int, int),
) {
    var wg sync.WaitGroup
    errors := make(chan error, numGoroutines)
    
    for i := 0; i < numGoroutines; i++ {
        wg.Add(1)
        go func(id int) {
            defer wg.Done()
            
            for j := 0; j < iterationsPerGoroutine; j++ {
                testFunc(id, j)
            }
        }(i)
    }
    
    // Wait for completion or errors
    done := make(chan struct{})
    go func() {
        wg.Wait()
        close(done)
    }()
    
    select {
    case err := <-errors:
        t.Errorf("Concurrent test failed: %v", err)
    case <-done:
        // Test passed
    }
}
```

## 8. Concurrency Prevention Checklist

### Development Guidelines
- [ ] Use `SafeObservable` when multiple goroutines call `observer.Next()`
- [ ] Always check context before operations in long-running goroutines
- [ ] Protect shared state with mutexes or atomic operations
- [ ] Don't share state between the subscriptions to an observable
- [ ] Provide proper cleanup in teardown functions

### Testing Requirements
- [ ] Run tests with `-race` flag regularly
- [ ] Include stress tests with high concurrency
- [ ] Test cancellation scenarios
- [ ] Verify resource cleanup under concurrent load
- [ ] Monitor goroutine counts in tests

### Code Review Points
- [ ] Check for unsynchronized shared state
- [ ] Verify context propagation through pipelines
- [ ] Look for potential deadlock scenarios
- [ ] Ensure proper error handling in concurrent code
- [ ] Confirm all goroutines have exit conditions

### Production Monitoring
- [ ] Monitor goroutine counts over time
- [ ] Track error rates
- [ ] Set up alerts for high concurrency
- [ ] Log race condition warnings
- [ ] Profile concurrent performance regularly

## Next Steps

- [Memory Leaks](./memory-leaks) - Memory leak detection and prevention
- [Performance Issues](./performance) - Performance optimization techniques
- [Debugging Techniques](./debugging) - Systematic debugging approaches
