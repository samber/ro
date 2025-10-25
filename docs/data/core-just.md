---
name: Just
slug: just
sourceRef: operator_creation.go#L40
type: core
category: creation
signatures:
  - "func Just[T any](values ...T)"
  - "func Of[T any](values ...T)"
playUrl: https://go.dev/play/p/2CTim8maLwZ
variantHelpers:
  - core#creation#just
  - core#creation#of
similarHelpers:
  - core#creation#fromslice
  - core#creation#empty
position: 0
---

Creates an Observable that emits a specific sequence of values.

```go
obs := ro.Just(1, 2, 3, 4, 5)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Completed
```

### Just with no values

```go
obs := ro.Just[int]() // Equivalent to ro.Empty[int]()

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed
```

### Just with complex types

```go
type Person struct {
    Name string
    Age  int
}

obs := ro.Just(
    Person{"Alice", 25},
    Person{"Bob", 30},
    Person{"Charlie", 35},
)

sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()

// Next: {Alice 25}
// Next: {Bob 30}
// Next: {Charlie 35}
// Completed
```
