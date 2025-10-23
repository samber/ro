---
name: SubscribeOn
slug: subscribeon
sourceRef: scheduler.go#L59
type: core
category: utility
signatures:
  - "func SubscribeOn[T any](bufferSize int)"
playUrl:
variantHelpers:
  - core#utility#subscribeon
similarHelpers: [core#utility#observeon]
position: 95
---

Schedule the upstream flow to a different goroutine. This detaches the subscription from the current goroutine and processes emissions in a separate goroutine.

```go
obs := ro.Pipe[string, string](
    ro.Just("main", "thread"),
    ro.SubscribeOn(10), // Process in background goroutine with buffer size 10
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: main (processed in background goroutine)
// Next: thread (processed in background goroutine)
// Completed
```

### With heavy computations

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Map(func(i int) int {
        // Simulate heavy computation
        time.Sleep(100 * time.Millisecond)
        return i * i
    }),
    ro.SubscribeOn(100), // Large buffer for heavy computations
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 4, 9, 16, 25 (computed in background goroutine)
// Completed
```

### With backpressure control

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(10*time.Millisecond), // Fast producer
    ro.SubscribeOn(5), // Small buffer causes backpressure
)

sub := obs.Subscribe(ro.NewObserver[int64](
    func(value int64) {
        time.Sleep(50 * time.Millisecond) // Slow consumer
        fmt.Printf("Next: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: values processed in background goroutine with backpressure
// Completed
```

### With error handling

```go
obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("background error")),
    ro.SubscribeOn(10),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: background error
```

### With multiple operators

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c"),
    ro.Map(func(s string) string { return strings.ToUpper(s) }),
    ro.Filter(func(s string) bool { return s != "B" }),
    ro.SubscribeOn(20), // All upstream operations in background
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: A
// Next: C
// Completed
```