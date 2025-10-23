---
name: Distinct
slug: distinct
sourceRef: operator_filter.go#L73
type: core
category: filtering
signatures:
  - "func Distinct[T comparable]()"
playUrl:
variantHelpers:
  - core#filtering#distinct
similarHelpers:
  - core#filtering#distinctby
position: 60
---

Returns an observable sequence that contains only distinct elements.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 2, 3, 1, 4, 3, 5),
    ro.Distinct[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Completed
```

### With strings

```go
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "apple", "cherry", "banana"),
    ro.Distinct[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Next: banana
// Next: cherry
// Completed
```

### With custom types

```go
type Person struct {
    ID   int
    Name string
}

obs := ro.Pipe[Person, Person](
    ro.Just(
        Person{1, "Alice"},
        Person{2, "Bob"},
        Person{1, "Alice"}, // Duplicate
        Person{3, "Charlie"},
    ),
    ro.Distinct[Person](),
)

sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()

// Next: {1 Alice}
// Next: {2 Bob}
// Next: {3 Charlie}
// Completed
```

### Empty sequence

```go
obs := ro.Pipe[int, int](
    ro.Empty[int](),
    ro.Distinct[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```