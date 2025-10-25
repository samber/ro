---
name: Clamp
slug: clamp
sourceRef: operator_math.go#L197
type: core
category: math
signatures:
  - "func Clamp[T constraints.Ordered](min, max T)"
playUrl: https://go.dev/play/p/fu8O-BixXPM
variantHelpers:
  - core#math#clamp
similarHelpers: []
position: 150
---

Clamps values to be within a specified range.

```go
obs := ro.Pipe[int, int](
    ro.Just(-5, 0, 5, 10, 15),
    ro.Clamp(0, 10),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 0
// Next: 0
// Next: 5
// Next: 10
// Next: 10
// Completed
```

### With floats

```go
obs := ro.Pipe[float64, float64](
    ro.Just(-1.5, 0.0, 0.5, 1.0, 1.5),
    ro.Clamp(0.0, 1.0),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.0
// Next: 0.0
// Next: 0.5
// Next: 1.0
// Next: 1.0
// Completed
```

### With negative values

```go
obs := ro.Pipe[int, int](
    ro.Just(-20, -10, 0, 10, 20),
    ro.Clamp(-15, 15),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: -15
// Next: -10
// Next: 0
// Next: 10
// Next: 15
// Completed
```

### Edge case: min equals max

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Clamp(3, 3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Next: 3
// Next: 3
// Next: 3
// Next: 3
// Completed
```