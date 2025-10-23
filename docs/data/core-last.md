---
name: Last
slug: last
sourceRef: operator_filter.go#L620
type: core
category: filtering
signatures:
  - "func Last[T any](predicate func(item T) bool)"
  - "func LastWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool))"
  - "func LastI[T any](predicate func(item T, index int64) bool)"
  - "func LastIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool))"
playUrl:
variantHelpers:
  - core#filtering#last
  - core#filtering#lastwithcontext
  - core#filtering#lasti
  - core#filtering#lastiwithcontext
similarHelpers:
  - core#filtering#first
  - core#filtering#tail
  - core#filtering#takelast
position: 40
---

Emits only the last item (or the last item that satisfies a predicate) from an Observable sequence.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Last[int](func(i int) bool {
        return true // Match all items, returns last
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 5
// Completed
```

### Last with predicate

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Last[int](func(i int) bool {
        return i < 4 // Find last item less than 4
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.LastWithContext(func(ctx context.Context, i int) (context.Context, bool) {
        return ctx, i < 4
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.LastI(func(i int, index int64) bool {
        return index < 3 // Find last item with index < 3
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
    ro.LastIWithContext(func(ctx context.Context, i int, index int64) (context.Context, bool) {
        return ctx, i < 4 && index < 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Completed
```

### Edge case: No matching items

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Last[int](func(i int) bool {
        return i > 10 // No items match
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// No Next values, completes without emitting
// Completed
```

### Edge case: Single item

```go
obs := ro.Pipe[int, int](
    ro.Just(42),
    ro.Last[int](func(i int) bool {
        return i > 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 42
// Completed
```