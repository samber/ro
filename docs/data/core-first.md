---
name: First
slug: first
sourceRef: operator_filter.go#L566
type: core
category: filtering
signatures:
  - "func First[T any](predicate func(item T) bool)"
  - "func FirstWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool))"
  - "func FirstI[T any](predicate func(item T, index int64) bool)"
  - "func FirstIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool))"
playUrl: https://go.dev/play/p/yneVKit6vh0
variantHelpers:
  - core#filtering#first
  - core#filtering#firstwithcontext
  - core#filtering#firsti
  - core#filtering#firstiwithcontext
similarHelpers:
  - core#filtering#last
  - core#filtering#head
  - core#filtering#take
position: 30
---

Emits only the first item (or the first item that satisfies a predicate) from an Observable sequence.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.First[int](func(i int) bool {
        return true // Match all items, returns first
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Completed
```

### First with predicate

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.First[int](func(i int) bool {
        return i > 3 // Find first item greater than 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FirstWithContext(func(ctx context.Context, i int) (context.Context, bool) {
        return ctx, i > 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FirstI(func(i int, index int64) bool {
        return index > 2 // Find first item with index > 2
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4 (index 3)
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FirstIWithContext(func(ctx context.Context, i int, index int64) (context.Context, bool) {
        return ctx, i > 3 && index > 2
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```

### Edge case: No matching items

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.First[int](func(i int) bool {
        return i > 10 // No items match
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// No Next values, completes without emitting
// Completed
```