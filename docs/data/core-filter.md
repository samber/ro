---
name: Filter
slug: filter
sourceRef: operator_filter.go#L25
type: core
category: filtering
signatures:
  - "func Filter[T any](predicate func(item T) bool)"
  - "func FilterWithContext[T any](predicate func(ctx context.Context, item T) (context.Context, bool))"
  - "func FilterI[T any](predicate func(item T, index int64) bool)"
  - "func FilterIWithContext[T any](predicate func(ctx context.Context, item T, index int64) (context.Context, bool))"
playUrl:
variantHelpers:
  - core#filtering#filter
  - core#filtering#filterwithcontext
  - core#filtering#filteri
  - core#filtering#filteriwithcontext
similarHelpers: []
position: 0
---

Emits only those items from an Observable that pass a predicate test.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Filter(func(i int) bool {
        return i%2 == 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 2
// Next: 4
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FilterWithContext(func(ctx context.Context, i int) (context.Context, bool) {
        return ctx, i%2 == 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 2
// Next: 4
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FilterI(func(i int, index int64) bool {
        return index > 1 // Skip first two elements
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Next: 4
// Next: 5
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.FilterIWithContext(func(ctx context.Context, i int, index int64) (context.Context, bool) {
        return ctx, index > 1 && i%2 == 0
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Completed
```