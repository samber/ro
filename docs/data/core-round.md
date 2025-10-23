---
name: Round
slug: round
sourceRef: operator_math.go#L110
type: core
category: math
signatures:
  - "func Round()"
playUrl:
variantHelpers:
  - core#math#round
similarHelpers:
  - core#math#floor
  - core#math#ceil
  - core#math#trunc
position: 30
---

Emits the rounded values from the source Observable using standard mathematical rounding rules.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(1.1, 1.5, 1.9, 2.5, -1.5),
    ro.Round(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 1, 2, 2, 2, -2
// Completed
```

### With decimal precision

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.14159, 2.71828, 1.41421),
    ro.Round(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3, 3, 1
// Completed
```

### With negative numbers

```go
obs := ro.Pipe[float64, float64](
    ro.Just(-2.3, -2.7, -3.5),
    ro.Round(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: -2, -3, -4
// Completed
```

### With integer-like values

```go
obs := ro.Pipe[float64, float64](
    ro.Just(5.0, 6.0001, 4.9999),
    ro.Round(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 5, 6, 5
// Completed
```

### In data processing pipeline

```go
obs := ro.Pipe[int64, float64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](5),
    ro.Map(func(_ int64) float64 {
        return rand.Float64() * 100 // Random values 0-100
    }),
    ro.Round(), // Round to whole numbers
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
time.Sleep(600 * time.Millisecond)
sub.Unsubscribe()

// Random rounded integers between 0-100
// Example: 42, 87, 15, 93, 28
```

### With financial calculations

```go
obs := ro.Pipe[float64, float64](
    ro.Just(12.345, 67.890, 123.456),
    ro.Round(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 12, 68, 123
// Completed
```