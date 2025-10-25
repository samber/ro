---
name: TakeWhile
slug: takewhile
sourceRef: operator_filter.go#L339
type: core
category: filtering
signatures:
  - "func TakeWhile[T any](predicate func(item T) bool)"
  - "func TakeWhileWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool))"
  - "func TakeWhileI[T any](predicate func(item T, index int64) bool)"
  - "func TakeWhileIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool))"
playUrl: https://go.dev/play/p/lxV03GzOa2J
variantHelpers:
  - core#filtering#takewhile
  - core#filtering#takewhilewithcontext
  - core#filtering#takewhilei
  - core#filtering#takewhileiwithcontext
similarHelpers:
  - core#filtering#take
  - core#filtering#takelast
  - core#filtering#takeuntil
  - core#filtering#head
position: 20
---

Emits items emitted by an Observable so long as a specified condition is true, then completes.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.TakeWhile(func(i int) bool {
        return i < 5
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.TakeWhileWithContext(func(ctx context.Context, i int) (context.Context, bool) {
        return ctx, i < 5
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.TakeWhileI(func(i int, index int64) bool {
        return index < 3 // Take first 4 elements (index 0, 1, 2, 3)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.TakeWhileIWithContext(func(ctx context.Context, i int, index int64) (context.Context, bool) {
        return ctx, i < 5 && index < 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Completed
```

### Edge case: Never completes if condition always true

```go
// In practice, this would complete when the source completes
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.TakeWhile(func(i int) bool {
        return true // Always true
    }),
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

### Edge case: Completes immediately if condition always false

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.TakeWhile(func(i int) bool {
        return false // Always false
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed
```