---
name: Start
slug: start
sourceRef: operator_creation.go#L66
type: core
category: creation
signatures:
  - "func Start[T any](action func() T)"
playUrl: https://go.dev/play/p/Jz7oyagu07u
variantHelpers:
  - core#creation#start
similarHelpers:
  - core#creation#defer
  - core#creation#future
position: 36
---

Creates an Observable that emits the result of an action function for each subscriber, running the action synchronously.

```go
obs := ro.Start(func() int {
    // perform work such as HTTP request, database query...
    return 42
})

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 42
// Completed
```

### With complex computation

```go
obs := ro.Start(func() string {
    time.Sleep(50 * time.Millisecond) // Simulate work
    return fmt.Sprintf("Result: %d", 21*2)
})

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Result: 42" (after ~50ms)
// Completed
```

### Fresh execution for each subscriber

```go
obs := ro.Start(func() int {
    return rand.Intn(100) // Different random number each time
})

sub1 := obs.Subscribe(ro.PrintObserver[int]())
sub2 := obs.Subscribe(ro.PrintObserver[int]())

defer sub1.Unsubscribe()
defer sub2.Unsubscribe()

// Each subscriber gets a fresh execution
// sub1 might get: 73
// sub2 might get: 15
```

### With error handling

```go
obs := ro.Start(func() string {
    file, err := os.Open("nonexistent.txt")
    if err != nil {
        panic(err) // This will cause the observable to error
    }
    defer file.Close()

    content, _ := io.ReadAll(file)
    return string(content)
})

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: open nonexistent.txt: no such file or directory
```

### For expensive calculations

```go
obs := ro.Start(func() []int {
    // Simulate expensive calculation
    result := make([]int, 0)
    for i := 0; i < 1000; i++ {
        result = append(result, i*i)
    }
    return result
})

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [0, 1, 4, 9, 16, ..., 998001] (after calculation completes)
// Completed
```

### With external dependencies

```go
obs := ro.Start(func() time.Time {
    return time.Now() // Current time at execution
})

sub1 := obs.Subscribe(ro.PrintObserver[time.Time]())
time.Sleep(10 * time.Millisecond)
sub2 := obs.Subscribe(ro.PrintObserver[time.Time]())

defer sub1.Unsubscribe()
defer sub2.Unsubscribe()

// Each subscriber gets the time when their subscription started
// Different times for sub1 and sub2
```