---
name: DefaultIfEmpty
slug: defaultifempty
sourceRef: operator_conditional.go#L187
type: core
category: conditional
signatures:
  - "func DefaultIfEmpty[T any](defaultValue T)"
  - "func DefaultIfEmptyWithContext[T any](defaultValue T)"
playUrl:
variantHelpers:
  - core#conditional#defaultifempty
  - core#conditional#defaultifemptywithcontext
similarHelpers: []
position: 40
---

Emits a default value if the source observable completes without emitting any items.

```go
obs := ro.Pipe[int, int](
    ro.Empty[int](),
    ro.DefaultIfEmpty(42),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 42
// Completed
```

### With non-empty source

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.DefaultIfEmpty(42),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With context

```go
obs := ro.Pipe[string, string](
    ro.Empty[string](),
    ro.DefaultIfEmptyWithContext("default value"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: default value
// Completed
```

### With filtered empty result

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Filter(func(i int) bool {
        return i > 10 // No items match
    }),
    ro.DefaultIfEmpty(-1),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: -1
// Completed
```