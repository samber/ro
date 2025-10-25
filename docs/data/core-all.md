---
name: All
slug: all
sourceRef: operator_conditional.go#L24
type: core
category: conditional
signatures:
  - "func All[T any](predicate func(item T) bool)"
  - "func AllWithContext[T any](predicate func(ctx context.Context, item T) bool)"
  - "func AllI[T any](predicate func(item T, index int64) bool)"
  - "func AllIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool)"
playUrl: https://go.dev/play/p/t22F_crlA-l
variantHelpers:
  - core#conditional#all
  - core#conditional#allwithcontext
  - core#conditional#alli
  - core#conditional#alliwithcontext
similarHelpers: []
position: 0
---

Determines whether all elements of an observable sequence satisfy a condition.

```go
obs := ro.Pipe[int, bool](
    ro.Just(1, 2, 3, 4, 5),
    ro.All(func(i int) bool {
        return i > 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

### With context

```go
obs := ro.Pipe[int, bool](
    ro.Just(1, 2, 3, 4, 5),
    ro.AllWithContext(func(ctx context.Context, n int) bool {
        return n > 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

### With index

```go
obs := ro.Pipe[int, bool](
    ro.Just(1, 2, 3, 4, 5),
    ro.AllI(func(n int, index int64) bool {
        return index < 3 // Only check first 3 elements
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, bool](
    ro.Just(1, 2, 3, 4, 5),
    ro.AllIWithContext(func(ctx context.Context, n int, index int64) bool {
        return n > 0 && index < 4
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```