---
title: 🔍 Debugging Techniques
description: Systematic approaches to debugging reactive streams
sidebar_position: 3
---

# 🔍 Debugging Techniques

Debugging reactive streams requires different approaches than traditional imperative code. This guide covers systematic techniques for identifying and resolving issues in `samber/ro` applications.

## 1. Stream Inspection with Tap Operators

The most effective way to debug streams is to add inspection points using Tap operators.

### Basic Stream Debugging

```go
// Add Tap operators to see what's happening at each step
func debugPipeline(source ro.Observable[int]) ro.Observable[string] {
    return ro.Pipe3(
        source,
        ro.TapOnNext(func(v int) { 
            log.Printf("🔵 Source emitted: %v", v) 
        }),
        ro.Map(func(x int) int { return x * 2 }),
        ro.TapOnNext(func(v int) { 
            log.Printf("🟡 After Map: %v", v) 
        }),
        ro.Filter(func(x int) bool { return x > 5 }),
        ro.TapOnNext(func(v int) { 
            log.Printf("🟢 After Filter: %v", v) 
        }),
        ro.Map(func(x int) string { return fmt.Sprintf("result-%d", x) }),
    )
}

// Usage with error and completion logging
debugPipeline(ro.Just(1, 2, 3, 4, 5, 6)).Subscribe(
    ro.NewObserver(
        func(v string) { log.Printf("✅ Final result: %v", v) },
        func(err error) { log.Printf("❌ Error: %v", err) },
        func() { log.Printf("🏁 Completed") },
    ),
)

// Or use the builtin debugger:
debugPipeline(ro.Just(1, 2, 3, 4, 5, 6)).Subscribe(
    ro.PrintObserver[string](),
)
```

### Conditional Debugging

```go
// Debug only specific values
func debugConditional[T any](predicate func(T) bool, message string) func(ro.Observable[T]) ro.Observable[T] {
    return ro.TapOnNext(func(v T) {
        if predicate(v) {
            log.Printf("🐛 DEBUG [%s]: %v", message, v)
        }
    })
}

// Usage: Debug only large numbers
pipeline := ro.Pipe2(
    ro.Just(1, 100, 2, 200, 3, 300),
    debugConditional(func(x int) bool { return x > 50 }, "large-numbers"),
    ro.Map(func(x int) int { return x / 2 }),
)
```

### Debug with a custom operator

```go
// Reusable debug operator
func DebugStream[T any](name string, logValues bool, logErrors bool, logCompletion bool) func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewObservable(func(observer ro.Observer[T]) ro.Teardown {
            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    if logValues {
                        log.Printf("📡 [%s] Next: %v", name, value)
                    }
                    observer.Next(value)
                },
                func(err error) {
                    if logErrors {
                        log.Printf("💥 [%s] Error: %v", name, err)
                    }
                    observer.Error(err)
                },
                func() {
                    if logCompletion {
                        log.Printf("✨ [%s] Complete", name)
                    }
                    observer.Complete()
                },
            ))
            return sub.Unsubscribe
        })
    }
}

// Usage
pipeline := ro.Pipe3(
    ro.Just(1, 2, 3),
    DebugStream("input", true, true, true),
    ro.Map(func(x int) int { return x * 2 }),
    DebugStream("mapped", true, true, false), // No completion logging
)
```

## 2. Test-Driven Debugging

Isolate problematic components by testing them individually.

### Unit Testing Individual Operators

```go
func TestProblematicOperator(t *testing.T) {
    // Test with known input
    input := ro.Just(1, 2, 3, 4, 5)
    obs := yourOperator()(input)
    
    // Collect results
    values, err := ro.Collect(obs)
    require.NoError(t, err)
    
    // Verify expectations
    expected := []int{2, 4, 6, 8, 10}
    assert.Equal(t, expected, values)
}

func TestErrorHandling(t *testing.T) {
    // Test with error-producing input
    input := ro.Throw[int](fmt.Errorf("test error"))
    obs := yourOperator()(input)
    
    // Should handle the error gracefully
    values, err := ro.Collect(obs)
    assert.Error(t, err)
    assert.Empty(t, values)
}

func TestEmptySource(t *testing.T) {
    // Test with error-producing input
    input := ro.Empty[int]()
    obs := yourOperator()(input)

    // Collect results
    values, err := ro.Collect(obs)
    require.NoError(t, err)
    
    // Verify expectations
    assert.Equal(t, []int{}, values)
}

func TestBlockedSource(t *testing.T) {
    // Test with error-producing input
    input := ro.Never[int]()
    obs := yourOperator()(input)

    // Collect results
    ctx, _ := context.WithTimeout(context.Background(), 5*time.Millisecond)
    sub := obs.SubscribeWithContext(ctx, ro.PrintObserver[int]())   // 💥 blocking

    t.False(t, sub.IsClosed())
    time.Sleep(10*time.Millisecond)
    t.True(t, sub.IsClosed())
}
```

## 3. Go Tooling Integration

### Race Detection

```bash
# Run tests with race detector
go test -race ./...

# Run your application with race detector
go run -race main.go
```

```go
// Test that demonstrates potential race condition
func TestConcurrentAccess(t *testing.T) {
    observable := ro.Just(1, 2, 3, 4, 5)
    
    var wg sync.WaitGroup
    for i := 0; i < 10; i++ {
        wg.Add(1)
        go func() {
            defer wg.Done()
            values, err := ro.Collect(observable)
            require.NoError(t, err)
            t.Logf("Goroutine %d got: %v", i, values)
        }()
    }
    
    wg.Wait()
}
```

### Memory Profiling

```bash
# Generate CPU profile
go test -cpuprofile=cpu.prof -bench=.

# Generate memory profile
go test -memprofile=mem.prof -bench=.

# Analyze profiles
go tool pprof cpu.prof
go tool pprof mem.prof
```

```go
func BenchmarkYourOperator(b *testing.B) {
    source := ro.Just(1, 2, 3, 4, 5)
    operator := YourOperator()
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        values, err := ro.Collect(operator(source))
        if err != nil {
            b.Fatal(err)
        }
        _ = values
    }
}
```

### Runtime Tracing

```go
func EnableTracing() {
    // Enable runtime tracing
    trace.Start(os.Stdout)
    defer trace.Stop()
    
    // Your reactive code here
    source := ro.Just(1, 2, 3, 4, 5)
    obs := yourOperator()(source)
    
    values, err := ro.Collect(obs)
    if err != nil {
        log.Printf("Error: %v", err)
    }
    log.Printf("Result: %v", values)
}
```

## 4. Custom Debugging Operators

### Value History Tracking

```go
type ValueTracker[T any] struct {
    mu     sync.Mutex
    values []T
    errors []error
}

func (t *ValueTracker[T]) Track() func(ro.Observable[T]) ro.Observable[T] {
    return func(source ro.Observable[T]) ro.Observable[T] {
        return ro.NewObservable(func(observer ro.Observer[T]) ro.Teardown {
            sub := source.Subscribe(ro.NewObserver(
                func(value T) {
                    t.mu.Lock()
                    t.values = append(t.values, value)
                    t.mu.Unlock()
                    observer.Next(value)
                },
                func(err error) {
                    t.mu.Lock()
                    t.errors = append(t.errors, err)
                    t.mu.Unlock()
                    observer.Error(err)
                },
                observer.Complete,
            ))
            return sub.Unsubscribe
        })
    }
}

func (t *ValueTracker[T]) GetHistory() ([]T, []error) {
    t.mu.Lock()
    defer t.mu.Unlock()
    return append([]T(nil), t.values...), append([]error(nil), t.errors...)
}

// Usage
tracker := &ValueTracker[int]{}
pipeline := ro.Pipe4(
    ro.Just(1, 2, 3),
    ro.Filter(...),
    tracker.Track(),
    ro.Map(...),
    ro.Take(42),
)

values, err := ro.Collect(pipeline)
fmt.Printf("Final values: %v\n", values)
fmt.Printf("Tracked history: %v\n", tracker.GetHistory())
```

## 5. Debugging Checklist

When debugging reactive streams, follow this systematic approach:

### Step 1: Verify Basic Flow
- [ ] Add Tap operators to identify where values stop flowing
- [ ] Check if observable is hot vs cold as expected
- [ ] Verify subscription count and timing

### Step 2: Check Error Handling
- [ ] Ensure all observers handle errors
- [ ] Verify error propagation through pipeline
- [ ] Check for panic recovery behavior

### Step 3: Examine Context Usage
- [ ] Verify context propagation through all operators
- [ ] Check for unexpected cancellations
- [ ] Validate timeout and deadline behavior

### Step 4: Profile Resources
- [ ] Run with race detector (`go test -race`)
- [ ] Profile memory usage (`go test -memprofile`)
- [ ] Profile CPU usage (`go test -cpuprofile`)

### Step 5: Isolate Components
- [ ] Test individual operators in isolation
- [ ] Replace complex sources with simple test data
- [ ] Build up complexity gradually

## Next Steps

- [Common Issues](./common-issues) - Specific problem solutions
- [Performance Issues](./performance) - Performance optimization
- [Memory Leaks](./memory-leaks) - Memory leak detection