---
name: Ceil
slug: ceil
sourceRef: operator_math.go#L268
type: core
category: math
signatures:
  - "func Ceil()"
playUrl:
variantHelpers:
  - core#math#ceil
similarHelpers:
  - core#math#floor
  - core#math#round
  - core#math#abs
position: 4
---

Emits the ceiling (rounded up) of each number emitted by the source Observable.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.2, 4.7, -2.3, -5.8, 0.0, 7.0),
    ro.Ceil(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 4
// Next: 5
// Next: -2
// Next: -5
// Next: 0
// Next: 7
// Completed
```

### With infinity values

```go
obs := ro.Pipe[float64, float64](
    ro.Just(math.Inf(-1), -42.7, math.Inf(1), 3.14),
    ro.Ceil(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: -Inf
// Next: -42
// Next: +Inf
// Next: 4
// Completed
```

### With NaN values

```go
obs := ro.Pipe[float64, float64](
    ro.Just(math.NaN(), 2.3, math.NaN(), -1.7),
    ro.Ceil(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: NaN
// Next: 3
// Next: NaN
// Next: -1
// Completed
```

### With time-based emissions

```go
obs := ro.Pipe[int64, float64](
    ro.Interval(100*time.Millisecond),
    ro.Map(func(i int64) float64 {
        return float64(i) * 0.7 // Emit 0, 0.7, 1.4, 2.1, 2.8...
    }),
    ro.Ceil(),
    ro.Take(5),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
time.Sleep(600 * time.Millisecond)
sub.Unsubscribe()

// Next: 0
// Next: 1
// Next: 2
// Next: 3
// Next: 3
// Completed
```