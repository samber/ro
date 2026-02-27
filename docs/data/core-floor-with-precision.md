---
name: FloorWithPrecision
slug: floor-with-precision
sourceRef: operator_math.go#L296
type: core
category: math
signatures:
  - "func FloorWithPrecision(precision int)"
variantHelpers:
  - core#math#floor-with-precision
similarHelpers:
  - core#math#floor
  - core#math#round
  - core#math#ceil
position: 3
---

Floors each value emitted by the source Observable after shifting the decimal point `precision` places to the right.
Positive precisions keep additional fractional digits while still rounding down, whereas negative precisions round to powers of ten.

Any integer precision is accepted. Large magnitudes rely on chunked `big.Float` arithmetic shared with `CeilWithPrecision`; extremely large positive precisions fall back to returning the original values, while sufficiently negative precisions yield `0` for non-negative inputs and `-Inf` for negative ones.
`math.NaN()` and `math.Inf()` inputs propagate as-is, matching `math.Floor` semantics.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.14159, 2.71828, -1.2345),
    ro.FloorWithPrecision(2),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3.14
// Next: 2.71
// Next: -1.24
// Completed
```

### Handling very small numbers

```go
obs := ro.Pipe[float64, float64](
    ro.Just(0.000123, -0.000987, math.Inf(1), math.NaN()),
    ro.FloorWithPrecision(4),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.0001
// Next: -0.001
// Next: +Inf
// Next: NaN
// Completed
```

### Rounding to powers of ten

```go
obs := ro.Pipe[float64, float64](
    ro.Just(123.45, -123.45),
    ro.FloorWithPrecision(-2),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 100
// Next: -200
// Completed
```
