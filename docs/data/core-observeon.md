---
name: ObserveOn
slug: observeon
sourceRef: scheduler.go#L81
type: core
category: utility
signatures:
  - "func ObserveOn[T any](bufferSize int)"
playUrl: https://go.dev/play/p/YJ__KPmGUJo
variantHelpers:
  - core#utility#observeon
similarHelpers: [core#utility#subscribeon]
position: 96
---

Schedule the downstream flow to a different goroutine. Converts a push-based Observable into a pullable stream with backpressure capabilities.

```go
obs := ro.Pipe[string, string](
    ro.Just("fast", "emissions"),
    ro.ObserveOn(2), // Small buffer size
)

sub := obs.Subscribe(ro.NewObserver[string](
    func(value string) {
        time.Sleep(200 * time.Millisecond) // Slow consumer
        fmt.Printf("Next: %s\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: fast (after 200ms delay)
// Next: emissions (after 200ms delay)
// Completed
```

### With backpressure control

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(10*time.Millisecond), // Fast producer
    ro.ObserveOn(5), // Small buffer for backpressure
)

sub := obs.Subscribe(ro.NewObserver[int64](
    func(value int64) {
        time.Sleep(100 * time.Millisecond) // Slow consumer
        fmt.Printf("Next: %d\n", value)
    },
))
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Next: values processed with backpressure control
// (Fast producer will be blocked when buffer is full)
```

### With large buffer

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.ObserveOn(100), // Large buffer
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 (processed in background goroutine)
// Completed
```

### With error propagation

```go
obs := ro.Pipe[string, string](
    ro.Pipe[string, string](
        ro.Just("will error"),
        ro.Throw[string](errors.New("propagated error")),
    ),
    ro.ObserveOn(10),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Error: propagated error
```

### Combined with SubscribeOn

```go
obs := ro.Pipe[string, string](
    ro.Just("background", "processing"),
    ro.SubscribeOn(10), // Upstream in background
    ro.Map(strings.ToUpper),
    ro.ObserveOn(5),    // Downstream in different background thread
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: BACKGROUND
// Next: PROCESSING
// Completed
```