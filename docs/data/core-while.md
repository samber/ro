---
name: While
slug: while
sourceRef: operator_error_handling.go#L349
type: core
category: error-handling
signatures:
  - "func While[T any](condition func() bool)"
  - "func WhileI[T any](condition func(index int64) bool)"
  - "func WhileWithContext[T any](condition func(context.Context) (context.Context, bool))"
  - "func WhileIWithContext[T any](condition func(context.Context, index int64) (context.Context, bool))"
playUrl: https://go.dev/play/p/hMj3DBVtp73
variantHelpers:
  - core#error-handling#while
  - core#error-handling#whilei
  - core#error-handling#whilewithcontext
  - core#error-handling#whileiwithcontext
similarHelpers:
  - core#error-handling#dowhile
position: 60
---

Repeats the source observable as long as the condition returns true. Unlike DoWhile, While checks the condition before each iteration.

```go
counter := 0
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.While(func() bool {
        counter++
        return counter <= 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3 (counter becomes 1, condition: 1 <= 3 = true)
// Next: 1, 2, 3 (counter becomes 2, condition: 2 <= 3 = true)
// Next: 1, 2, 3 (counter becomes 3, condition: 3 <= 3 = true)
// Completed (counter becomes 4, condition: 4 <= 3 = false)
```

### WhileI with index

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b"),
    ro.WhileI(func(index int64) bool {
        return index < 2 // Repeat twice (index 0 and 1)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "a", "b" (index 0)
// Next: "a", "b" (index 1)
// Completed (index 2, condition false)
```

### WhileWithContext with cancellation

```go
ctx, cancel := context.WithCancel(context.Background())

obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.WhileWithContext(func(ctx context.Context) (context.Context, bool) {
        select {
        case <-ctx.Done():
            return ctx, false
        default:
            return ctx, true // Continue repeating
        }
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())

// Cancel after some iterations
cancel()
defer sub.Unsubscribe()
```

### WhileIWithContext with index and context

```go
ctx := context.Background()
obs := ro.Pipe[string, string](
    ro.Just("test"),
    ro.WhileIWithContext(func(ctx context.Context, index int64) (context.Context, bool) {
        fmt.Printf("Checking iteration %d\n", index)
        return ctx, index < 3 // Repeat 3 times
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Checking iteration 0
// Next: "test"
// Checking iteration 1
// Next: "test"
// Checking iteration 2
// Next: "test"
// Completed
```

### Conditional data generation

```go
dataAvailable := true
obs := ro.Pipe[int, int](
    ro.Defer(func() Observable[int] {
        // Simulate data fetch
        if !dataAvailable {
            return ro.Empty[int]()
        }
        dataAvailable = rand.Intn(2) == 0 // Randomly set availability
        return ro.Just(rand.Intn(100))
    }),
    ro.While(func() bool {
        return dataAvailable
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Emits values while data is available
// Stops when dataAvailable becomes false
```

### Polling with timeout

```go
startTime := time.Now()
timeout := 2 * time.Second

obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        // Check if timeout reached
        if time.Since(startTime) > timeout {
            return ro.Empty[string]()
        }
        // Simulate checking for messages
        if rand.Intn(5) == 0 {
            return ro.Just("new message")
        }
        return ro.Empty[string]()
    }),
    ro.While(func() bool {
        return time.Since(startTime) <= timeout
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(2500 * time.Millisecond)
sub.Unsubscribe()

// Polls for messages for 2 seconds
// Emits "new message" when available
```

### Rate-limited processing

```go
processed := 0
maxItems := 10
obs := ro.Pipe[int, int](
    ro.Defer(func() Observable[int] {
        if processed >= maxItems {
            return ro.Empty[int]()
        }
        processed++
        return ro.Just(processed)
    }),
    ro.While(func() bool {
        return processed < maxItems
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
// Completed
```

### With external resource monitoring

```go
type ResourceMonitor struct {
    isActive bool
    count    int
}

monitor := &ResourceMonitor{isActive: true}
obs := ro.Pipe[string, string](
    ro.Defer(func() Observable[string] {
        if !monitor.isActive || monitor.count >= 5 {
            return ro.Empty[string]()
        }
        monitor.count++
        return ro.Just(fmt.Sprintf("Resource update %d", monitor.count))
    }),
    ro.While(func() bool {
        return monitor.isActive && monitor.count < 5
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "Resource update 1"
// Next: "Resource update 2"
// Next: "Resource update 3"
// Next: "Resource update 4"
// Next: "Resource update 5"
// Completed
```