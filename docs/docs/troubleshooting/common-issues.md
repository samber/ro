---
title: Common Issues
description: Solutions to frequently encountered problems
sidebar_position: 2
---

# 🐛 Common Issues

This guide covers the most frequently encountered issues when working with `samber/ro` and their solutions.

## 1. Not Receiving Values

### Problem: Observable emits no values

```go
// This seems like it should work, but no values are received
observable := ro.Pipe1(
    ro.Just(1, 2, 3),
    ro.Map(func(x int) int { return x * 2 }),
)

// ❌ No output
observable.Subscribe(ro.OnNext(func(x int) {
    fmt.Println(x) // Never called
}))
```

**Cause:** Using `ro.OnNext()` with a blocking observable. The observable completes synchronously before the observer can handle values.

**Solution:** Use a full observer or handle the blocking nature:

```go
// ✅ Solution 1: Use full observer
observable.Subscribe(ro.NewObserver(
    func(x int) { fmt.Println(x) },      // Next
    func(err error) { fmt.Println(err) }, // Error  
    func() { fmt.Println("Done") },      // Complete
))

// ✅ Solution 2: Use TapXXX operators in the middle of your stream
observable := ro.Pipe1(
    ro.Just(1, 2, 3),
    ro.Take[int64](5),
    ro.Map(func(x int) int { return x * 2 }),
    ro.TapOnNext(func(x int) {
        fmt.Println("Value: %d", n) // print debug
    }),
    ro.Map(func(x int64) string {
        return fmt.Sprintf("Tick: %d", x)
    }),
)
observable.Subscribe(...)
```

### Problem: Hot observable not sharing values

```go
// Expected both subscribers to see same values
hot := ro.Connectable(ro.Just(1, 2, 3))

sub1 := hot.Subscribe(ro.OnNext(func(x int) {
    fmt.Println("Sub1:", x)
}))
sub2 := hot.Subscribe(ro.OnNext(func(x int) {
    fmt.Println("Sub2:", x)
}))

// ❌ No output - forgot to connect
```

**Cause:** Connectable observables need to be explicitly connected.

**Solution:** Connect the observable:

```go
// ✅ Connect the observable
connection := hot.Connect()
// Output: Sub1: 1, Sub2: 1, Sub1: 2, Sub2: 2, Sub1: 3, Sub2: 3
```

## 2. Error Handling Issues

### Problem: Errors being lost

```go
// ❌ Error is silently ignored
riskyObservable := ro.Pipe1(
    ro.Just(1, 2, 3),
    ro.Map(func(x int) int { 
        if x == 2 { 
            panic("error!") // This gets lost 
        }
        return x * 2 
    }),
)

riskyObservable.Subscribe(ro.OnNext(func(x int) {
    fmt.Println(x) // Only sees: 1, then stops
}))
```

**Cause:** Using `ro.OnNext()` ignores error notifications.

**Solution:** Use proper error handling:

```go
// ✅ Handle errors properly
riskyObservable.Subscribe(ro.NewObserver(
    func(x int) { fmt.Println("Next:", x) },
    func(err error) { fmt.Println("Error:", err) }, // Catch the error
    func() { fmt.Println("Complete") },
))
```

### Problem: Operator doesn't handle returned errors

```go
// ❌ MapErr errors not handled
stream := ro.Pipe1(
    ro.Just(1, 2, 3),
    ro.MapErr(func(x int) (int, error) {
        if x == 2 { return 0, fmt.Errorf("bad number") }
        return x * 2, nil
    }),
)

stream.Subscribe(ro.OnNext(func(x int) {
    fmt.Println(x) // Panic: unhandled error
}))
```

**Solution:** Use operators that handle errors or add error handling:

```go
// ✅ Option 1: Use Catch operator
safeStream := ro.Pipe2(
    ro.Just(1, 2, 3),
    ro.MapErr(func(x int) (int, error) {
        if x == 2 {
            return 0, fmt.Errorf("bad number")
        }
        return x * 2, nil
    }),
    ro.Catch(func(err error) ro.Observable[int] {
        fmt.Println("Recovered:", err)
        return ro.Just(0) // Fallback value
    }),
)

// ✅ Option 2: Handle in observer
stream.Subscribe(ro.NewObserver(
    func(x int) { fmt.Println("Next:", x) },
    func(err error) { fmt.Println("Error:", err) },
    func() { fmt.Println("Complete") },
))
```

## 3. Context and Cancellation Issues

### Problem: Context cancellation not respected

```go
// ❌ Operator doesn't check context
func badOperator() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000; i++ {
            time.Sleep(10 * time.Second)
            observer.Next(i) // Ignores context cancellation
        }
        observer.Complete()
        return nil
    })
}
```

**Solution:** Check context in long-running operations:

```go
// ✅ Context-aware operator
func goodOperator(ctx context.Context) ro.Observable[int] {
    return ro.NewObservableWithContext(func(ctx context.Context, observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000; i++ {
            select {
            case <-ctx.Done():
                return // Respect cancellation
            default:
                observer.Next(i)
                time.Sleep(10 * time.Second)
            }
        }
        observer.Complete()
        return nil
    })
}
```

### Problem: Context not propagated through pipeline

```go
// ❌ Context lost in custom operator
brokenOperator := func(source ro.Observable[int]) ro.Observable[string] {
    return ro.NewUnsafeObservable(func(destination ro.Observer[string]) ro.Teardown {
        // Context not passed through!
        sub := source.Subscribe(ro.NewObserver(
            func(value int) {
                destination.Next(fmt.Sprintf("item-%d", value))
            },
            destination.Error,
            destination.Complete,
        ))
        return sub.Unsubscribe
    })
}
```

**Solution:** Use context-aware observable creation:

```go
// ✅ Proper context propagation
workingOperator := func(source ro.Observable[int]) ro.Observable[string] {
    return ro.NewUnsafeObservableWithContext(func(subscriberCtx context.Context, destination ro.Observer[string]) ro.Teardown {
        sub := source.SubscribeWithContext(subscriberCtx, ro.NewObserverWithContext(
            func(ctx context.Context, value int) {
                destination.NextWithContext(ctx, fmt.Sprintf("item-%d", value))
            },
            destination.ErrorWithContext,
            destination.CompleteWithContext,
        ))
        return sub.Unsubscribe
    })
}
```

## 4. Backpressure and Performance Issues

### Problem: Fast producer overwhelms slow consumer

```go
// ❌ No backpressure handling
func fastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        // put in memory every values up-front
        numbers := []int{}
        for i := 0; i < 1000000; i++ {
            numbers = append(numbers, rand.IntN(42))
        }

        for i := 0; i < len(numbers); i++ {
            observer.Next(numbers[i]) // Produces faster than consumer can handle
        }

        observer.Complete()
        return nil
    })
}

slowConsumer := fastProducer().Subscribe(ro.OnNext(func(x int) {
    time.Sleep(1 * time.Millisecond) // Slow processing
    fmt.Println(x)
}))
// Result: Memory usage explodes
```

**Solution:** Implement buffering or throttling:

```go
// ✅ Add backpressure with buffer
func fastProducer() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        for i := 0; i < 1000000; i++ {
            observer.Next(rand.IntN(42)) // Produces values just in time
        }

        observer.Complete()
        return nil
    })
}

slowConsumer := ro.Pipe2(
    fastProducer(),
    ro.ObserveOn(10),  // a few values will be accumulated without blocking source
    ro.Flatten(),
).Subscribe(ro.OnNext(func(x int) {
    time.Sleep(1 * time.Millisecond) // Slow processing
    fmt.Println(x)
}))
```

## 5. Memory and Resource Leaks

### Problem: Goroutine leak from unsubscription

```go
// ❌ Goroutine continues after unsubscription
func leakyOperator() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        go func() {
            ticker := time.NewTicker(100 * time.Millisecond)
            defer ticker.Stop()
            
            i := 0
            for {
                select {
                case <-ticker.C:
                    observer.Next(i)
                    i++
                }
                // No way to stop this goroutine!
            }
        }()
        
        return func() {
            // Cleanup doesn't stop the goroutine
        }
    })
}
```

**Solution:** Provide proper cleanup mechanism:

```go
// ✅ Proper goroutine cleanup
func nonLeakyOperator() ro.Observable[int] {
    return ro.NewObservable(func(observer ro.Observer[int]) ro.Teardown {
        ctx, cancel := context.WithCancel(context.Background())
        ticker := time.NewTicker(100 * time.Millisecond)
        
        go func() {
            defer ticker.Stop()
            i := 0
            
            for {
                select {
                case <-ctx.Done():
                    return // Goroutine exits on cancellation
                case <-ticker.C:
                    observer.Next(i)
                    i++
                }
            }
        }()
        
        return func() {
            cancel() // Stop the goroutine
        }
    })
}
```

## 6. Operator Chaining Issues

### Problem: Wrong operator variant

```go
// ❌ Using wrong variant
numbers := ro.Just([]int{1, 2, 3}, []int{4, 5, 6})

// Want to flatten, but using Map instead of Flatten
obs := ro.Pipe1(
    numbers,
    ro.Map(func(slice []int) int {
        return slice[0] // Only gets first element
    }),
)
// Output: 1, 4 (missing other values)
```

**Solution:** Use correct operator:

```go
// ✅ Use Flatten for nested collections
obs := ro.Pipe1(
    ro.Just([]int{1, 2, 3}, []int{4, 5, 6}),
    ro.Flatten[int](), // Flatten nested slices
)
// Output: 1, 2, 3, 4, 5, 6
```

## Quick Reference

| Issue                      | Symptom                             | Quick Fix                                     |
| -------------------------- | ----------------------------------- | --------------------------------------------- |
| No values from `ro.Just`   | Silent subscription                 | Use `ro.NewObserver` instead of `ro.OnNext`   |
| Hot observable not working | No output from multiple subscribers | Call `.Connect()`                             |
| Lost errors                | Stream stops silently               | Add error observer with `ro.NewObserver`      |
| Context ignored            | Long operations don't cancel        | Use `*WithContext` variants and check context |
| Memory explosion           | Fast producer, slow consumer        | Add buffering or throttling                   |
| Goroutine leak             | Memory grows over time              | Provide proper cleanup in teardown            |

## Next Steps

- [Debugging Techniques](./debugging) - Systematic debugging approaches
- [Performance Issues](./performance) - Optimization and profiling
- [Memory Leaks](./memory-leaks) - Detection and prevention