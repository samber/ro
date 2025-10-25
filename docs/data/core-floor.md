---
name: Floor
slug: floor
sourceRef: operator_math.go#L248
type: core
category: math
signatures:
  - "func Floor()"
playUrl: https://go.dev/play/p/UulGlomv9K5
variantHelpers:
  - core#math#floor
similarHelpers:
  - core#math#ceil
  - core#math#round
  - core#math#abs
position: 2
---

Emits the floor (rounded down) of each number emitted by the source Observable.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.7, 4.2, -2.3, -5.8, 0.0, 7.0),
    ro.Floor(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3
// Next: 4
// Next: -3
// Next: -6
// Next: 0
// Next: 7
// Completed
```

### With infinity values

```go
obs := ro.Pipe[float64, float64](
    ro.Just(math.Inf(-1), -42.7, math.Inf(1), 3.14),
    ro.Floor(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: -Inf
// Next: -43
// Next: +Inf
// Next: 3
// Completed
```

### With NaN values

```go
obs := ro.Pipe[float64, float64](
    ro.Just(math.NaN(), 2.3, math.NaN(), -1.7),
    ro.Floor(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: NaN
// Next: 2
// Next: NaN
// Next: -2
// Completed
```

### With time-based emissions

```go
obs := ro.Pipe[int64, float64](
    ro.Interval(100*time.Millisecond),
    ro.Map(func(i int64) float64 {
        return float64(i) * 0.7 // Emit 0, 0.7, 1.4, 2.1, 2.8...
    }),
    ro.Floor(),
    ro.Take(5),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
time.Sleep(600 * time.Millisecond)
sub.Unsubscribe()

// Next: 0
// Next: 0
// Next: 1
// Next: 2
// Next: 2
// Completed
```