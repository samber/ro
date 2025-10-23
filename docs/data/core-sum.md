---
name: Sum
slug: sum
sourceRef: operator_math.go#L86
type: core
category: math
signatures:
  - "func Sum[T Numeric]()"
playUrl:
variantHelpers:
  - core#math#sum
similarHelpers:
  - core#math#average
  - core#math#count
  - core#math#reduce
position: 20
---

Calculates the sum of all values emitted by an Observable sequence and emits the total sum when the source completes.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Sum[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 15
// Completed
```

### Sum with floating point numbers

```go
obs := ro.Pipe[float64, float64](
    ro.Just(1.5, 2.5, 3.5),
    ro.Sum[float64](),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 7.5
// Completed
```

### Sum with negative numbers

```go
obs := ro.Pipe[int, int](
    ro.Just(10, -5, 3, -2),
    ro.Sum[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 6
// Completed
```

### Sum with single value

```go
obs := ro.Pipe[int, int](
    ro.Just(42),
    ro.Sum[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 42
// Completed
```

### Sum with empty observable

```go
obs := ro.Pipe[int, int](
    ro.Empty[int](),
    ro.Sum[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no Next values emitted)
```