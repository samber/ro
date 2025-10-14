---
name: Min
slug: min
sourceRef: operator_math.go#L133
type: core
category: math
signatures:
  - "func Min[T constraints.Ordered]()"
playUrl:
variantHelpers:
  - core#math#min
similarHelpers: []
position: 130
---

Finds the minimum value in an observable sequence.

```go
obs := ro.Pipe(
    ro.Just(5, 3, 8, 1, 4),
    ro.Min[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Completed
```

### With floats

```go
obs := ro.Pipe(
    ro.Just(3.14, 2.71, 1.61, 0.99),
    ro.Min[float64](),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.99
// Completed
```

### With strings

```go
obs := ro.Pipe(
    ro.Just("zebra", "apple", "banana", "cherry"),
    ro.Min[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Completed
```

### Empty sequence handling

```go
obs := ro.Pipe(
    ro.Empty[int](),
    ro.Min[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```