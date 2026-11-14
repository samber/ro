---
name: RoundWithPrecision
slug: round-with-precision
sourceRef: operator_math.go#L184
type: core
category: math
signatures:
  - "func RoundWithPrecision(precision int)"
variantHelpers:
  - core#math#round-with-precision
similarHelpers:
  - core#math#round
  - core#math#floor-with-precision
  - core#math#ceil
position: 32
---

Rounds each value emitted by the source Observable to the given number of decimal places.

Rounding is "half away from zero" (the rule used by `math.Round`) and is applied to the shortest decimal representation of each value, so results match the decimal literal written by the caller. For example, `0.145` is stored in binary as a value slightly below `0.145`, but `RoundWithPrecision(2)` emits `0.15`.

Positive precisions round to that many digits to the right of the decimal point, zero behaves like `Round()`, and negative precisions round to powers of ten. `math.NaN()` and `math.Inf()` inputs propagate unchanged, and a rounded finite result that exceeds the float64 range is emitted as `±Inf`. Precisions with an absolute value above 400 exceed float64's decimal resolution: larger positive precisions return values unchanged, while larger negative precisions emit `0` for finite inputs.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.14159, 2.71828, -1.2345),
    ro.RoundWithPrecision(2),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3.14
// Next: 2.72
// Next: -1.23
// Completed
```

### Rounding the decimal representation

```go
obs := ro.Pipe[float64, float64](
    ro.Just(0.145, 2.675, 1.005),
    ro.RoundWithPrecision(2),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.15
// Next: 2.68
// Next: 1.01
// Completed
```

### Rounding to powers of ten

```go
obs := ro.Pipe[float64, float64](
    ro.Just(123.45, 125, -125),
    ro.RoundWithPrecision(-1),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 120
// Next: 130
// Next: -130
// Completed
```
