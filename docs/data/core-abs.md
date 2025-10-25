---
name: Abs
slug: abs
sourceRef: operator_math.go#L228
type: core
category: math
signatures:
  - "func Abs()"
playUrl: https://go.dev/play/p/WCzxrucg7BC
variantHelpers:
  - core#math#abs
similarHelpers:
  - core#math#ceil
  - core#math#floor
  - core#math#round
position: 0
---

Emits the absolute value of each number emitted by the source Observable.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(-3.5, 2.1, -7.8, 0.0, 5.3),
    ro.Abs(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3.5
// Next: 2.1
// Next: 7.8
// Next: 0.0
// Next: 5.3
// Completed
```

### With time-based emissions

```go
obs := ro.Pipe[int64, float64](
    ro.Interval(100*time.Millisecond),
    ro.Map(func(i int64) float64 {
        return float64(i-5) // Emit -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5...
    }),
    ro.Abs(),
    ro.Take(5),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
time.Sleep(600 * time.Millisecond)
sub.Unsubscribe()

// Next: 5
// Next: 4
// Next: 3
// Next: 2
// Next: 1
// Completed
```

### With negative infinity

```go
obs := ro.Pipe[float64, float64](
    ro.Just(math.Inf(-1), -42.0, math.Inf(1)),
    ro.Abs(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: +Inf
// Next: 42
// Next: +Inf
// Completed
```