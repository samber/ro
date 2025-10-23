---
name: Max
slug: max
sourceRef: operator_math.go#L167
type: core
category: math
signatures:
  - "func Max[T constraints.Ordered]()"
playUrl:
variantHelpers:
  - core#math#max
similarHelpers: []
position: 140
---

Finds the maximum value in an observable sequence.

```go
obs := ro.Pipe[int, int](
    ro.Just(5, 3, 8, 1, 4),
    ro.Max[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 8
// Completed
```

### With floats

```go
obs := ro.Pipe[float64, float64](
    ro.Just(3.14, 2.71, 1.61, 0.99),
    ro.Max[float64](),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3.14
// Completed
```

### With strings

```go
obs := ro.Pipe[string, string](
    ro.Just("zebra", "apple", "banana", "cherry"),
    ro.Max[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: zebra
// Completed
```

### With custom types

```go
type Person struct {
    Name string
    Age  int
}

obs := ro.Pipe[Person, int](
    ro.Just(
        Person{"Alice", 25},
        Person{"Bob", 30},
        Person{"Charlie", 20},
    ),
    ro.Map[Person, int](func(p Person) int { return p.Age }),
    ro.Max[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 30
// Completed
```