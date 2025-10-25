---
name: Find
slug: find
sourceRef: operator_conditional.go#L126
type: core
category: conditional
signatures:
  - "func Find[T any](predicate func(item T) bool)"
  - "func FindWithContext[T any](predicate func(ctx context.Context, item T) bool)"
  - "func FindI[T any](predicate func(item T, index int64) bool)"
  - "func FindIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool)"
playUrl: https://go.dev/play/p/2f5rn0HoKeq
variantHelpers:
  - core#conditional#find
  - core#conditional#findwithcontext
  - core#conditional#findi
  - core#conditional#findiwithcontext
similarHelpers: []
position: 20
---

Finds the first element in an observable sequence that satisfies a condition.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Find(func(i int) bool {
        return i%2 == 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 2
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FindWithContext(func(ctx context.Context, n int) bool {
        return n > 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```

### With index

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c", "d", "e"),
    ro.FindI(func(item string, index int64) bool {
        return index >= 2 // Find item at position 2
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: c
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(10, 20, 30, 40, 50),
    ro.FindIWithContext(func(ctx context.Context, n int, index int64) bool {
        return n > 25 && index > 1
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 30
// Completed
```