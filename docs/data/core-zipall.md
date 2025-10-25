---
name: ZipAll
slug: zipall
sourceRef: operator_combining.go#L83
type: core
category: combining
signatures:
  - "func ZipAll[T any]()"
playUrl: https://go.dev/play/p/FcpgTItKX-Q
variantHelpers:
  - core#combining#zipall
similarHelpers:
  - core#combining#zip
  - core#combining#zipwith
  - core#combining#combinelatestall
position: 22
---

Creates an Observable that zips items from multiple Observable sources provided as a higher-order Observable, emitting arrays of zipped values.

```go
obs := ro.Pipe[Observable[any], []any](
    ro.Just(
        ro.Just(1, 2, 3),
        ro.Just("A", "B", "C"),
        ro.Just(true, false, true),
    ),
    ro.ZipAll[any](),
)

sub := obs.Subscribe(ro.PrintObserver[[]any]())
defer sub.Unsubscribe()

// Next: [1, A, true]
// Next: [2, B, false]
// Next: [3, C, true]
// Completed
```

### Dynamic observable collection

```go
// Create observables dynamically
observables := []Observable[int]{
    ro.Just(1, 2),
    ro.Just(3, 4),
    ro.Just(5, 6),
}

obs := ro.Pipe[Observable[int], []int](
    ro.Just(observables...),
    ro.ZipAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1, 3, 5]
// Next: [2, 4, 6]
// Completed
```

### With different types

```go
obs := ro.Pipe[Observable[any], []any](
    ro.Just(
        ro.Just(1, 2),
        ro.Just("A", "B"),
        ro.Just(true, false),
        ro.Just(1.1, 2.2),
    ),
    ro.ZipAll[any](),
)

sub := obs.Subscribe(ro.PrintObserver[[]any]())
defer sub.Unsubscribe()

// Next: [1, "A", true, 1.1]
// Next: [2, "B", false, 2.2]
// Completed
```

### Edge case: Empty observable of observables

```go
obs := ro.Pipe[Observable[int], []int](
    ro.Empty[Observable[int]](),
    ro.ZipAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// No items emitted, completes immediately
// Completed
```

### Edge case: Single observable in collection

```go
obs := ro.Pipe[Observable[int], []int](
    ro.Just(ro.Just(1, 2, 3)),
    ro.ZipAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1]
// Next: [2]
// Next: [3]
// Completed
```

### Edge case: Different length observables

```go
obs := ro.Pipe[Observable[any], []any](
    ro.Just(
        ro.Just(1, 2, 3, 4, 5), // 5 items
        ro.Just("A", "B"),       // 2 items
        ro.Just(true, false),    // 2 items
    ),
    ro.ZipAll[any](),
)

sub := obs.Subscribe(ro.PrintObserver[[]any]())
defer sub.Unsubscribe()

// Next: [1, "A", true]
// Next: [2, "B", false]
// Completed
// Only zips up to the shortest observable
```