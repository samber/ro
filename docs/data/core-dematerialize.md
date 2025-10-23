---
name: Dematerialize
slug: dematerialize
sourceRef: operator_utility.go#L488
type: core
category: utility
signatures:
  - "func Dematerialize[T any]()"
playUrl:
variantHelpers:
  - core#utility#dematerialize
similarHelpers:
  - core#utility#materialize
position: 470
---

Converts an observable of notifications back into an observable sequence.

```go
// Create notifications
notifications := []ro.Notification[int]{
    {Value: 1, HasValue: true},
    {Value: 2, HasValue: true},
    {Value: 3, HasValue: true},
    {Error: nil, HasValue: false}, // Completion
}

obs := ro.Pipe[ro.Notification[int], int](
    ro.FromSlice(notifications),
    ro.Dematerialize[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With error notifications

```go
// Create notifications including error
notifications := []ro.Notification[string]{
    {Value: "hello", HasValue: true},
    {Value: "world", HasValue: true},
    {Error: errors.New("test error"), HasValue: false},
}

obs := ro.Pipe[ro.Notification[string], string](
    ro.FromSlice(notifications),
    ro.Dematerialize[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello
// Next: world
// Error: test error
```

### Round trip with Materialize

```go
original := ro.Just(1, 2, 3)

// Materialize then dematerialize
roundTrip := ro.Pipe[int, int](
    original,
    ro.Materialize[int](),
    ro.Dematerialize[int](),
)

sub := roundTrip.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### Processing notifications manually

```go
// Create custom notification sequence
notifications := []ro.Notification[float64]{
    {Value: 3.14, HasValue: true},
    {Value: 2.71, HasValue: true},
    {Error: nil, HasValue: false},
}

obs := ro.Pipe[ro.Notification[float64], float64](
    ro.FromSlice(notifications),
    ro.Dematerialize[float64](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value float64) {
        fmt.Printf("Received: %.2f\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Sequence completed")
    },
))
defer sub.Unsubscribe()

// Received: 3.14
// Received: 2.71
// Sequence completed
```