---
name: Timer
slug: timer
sourceRef: operator_creation.go#L95
type: core
category: creation
signatures:
  - "func Timer(d time.Duration)"
playUrl:
variantHelpers:
  - core#creation#timer
similarHelpers:
  - core#creation#interval
  - core#creation#intervalwithinitial
position: 30
---

Creates an Observable that emits a single value (0) after a specified delay, then completes.

```go
obs := ro.Timer(1 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1500 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 1000ms)
// Completed
```

### Short delay

```go
obs := ro.Timer(100 * time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(200 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 100ms)
// Completed
```

### With other operators

```go
obs := ro.Pipe[int64, string](
    ro.Timer(500*time.Millisecond),
    ro.Map(func(_ int64) string {
        return "Timer fired!"
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: "Timer fired!" (after 500ms)
// Completed
```

### Timeout simulation

```go
timeout := ro.Timer(2 * time.Second)
dataSource := ro.Just("data")

// Race between timeout and data
raceResult := ro.Race(timeout, dataSource)

sub := raceResult.Subscribe(ro.PrintObserver[any]())
defer sub.Unsubscribe()

// If data emits before timeout: Next: "data", Completed
// If timeout fires first: Next: 0, Completed
```