---
name: Interval
slug: interval
sourceRef: operator_creation.go#L75
type: core
category: creation
signatures:
  - "func Interval(interval time.Duration)"
playUrl: https://go.dev/play/p/Lct91E7w17_B
variantHelpers:
  - core#creation#interval
similarHelpers:
  - core#creation#intervalwithinitial
  - core#creation#timer
  - core#creation#range
position: 20
---

Creates an Observable that emits sequential numbers every specified interval of time.

```go
obs := ro.Interval(100 * time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(550 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 100ms)
// Next: 1 (after 200ms)
// Next: 2 (after 300ms)
// Next: 3 (after 400ms)
// Next: 4 (after 500ms)
```

### Using Interval for periodic operations

```go
obs := ro.Interval(1 * time.Second)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(3500 * time.Millisecond)
sub.Unsubscribe()

// Next: 0 (after 1 second)
// Next: 1 (after 2 seconds)
// Next: 2 (after 3 seconds)
```

### Practical example: Heartbeat simulation

```go
ticker := ro.Interval(500 * time.Millisecond)
heartbeat := ro.Pipe[int64, string](ticker, ro.Map(func(i int64) string {
    return "❤️"
}))

sub := heartbeat.Subscribe(ro.PrintObserver[string]())
time.Sleep(2200 * time.Millisecond)
sub.Unsubscribe()

// Next: "❤️" (after 500ms)
// Next: "❤️" (after 1000ms)
// Next: "❤️" (after 1500ms)
// Next: "❤️" (after 2000ms)
```

### With Take for limited emissions

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.Take[int64](5),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 0
// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### Edge case: Very short interval

```go
obs := ro.Interval(1 * time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(10 * time.Millisecond)
sub.Unsubscribe()

// Will emit rapidly based on system scheduling
// Next: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9...
```