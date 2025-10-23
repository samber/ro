---
name: Map
slug: map
sourceRef: operator_transformations.go#L29
type: core
category: transformation
signatures:
  - "func Map[T any, R any](project func(item T) R)"
  - "func MapWithContext[T any, R any](project func(ctx context.Context, item T) (context.Context, R))"
  - "func MapI[T any, R any](project func(item T, index int64) R)"
  - "func MapIWithContext[T any, R any](project func(ctx context.Context, item T, index int64) (context.Context, R))"
playUrl: https://go.dev/play/p/JhTBEQFQGYr
variantHelpers:
  - core#transformation#map
  - core#transformation#mapwithcontext
  - core#transformation#mapi
  - core#transformation#mapiwithcontext
similarHelpers:
  - core#transformation#mapto,
  - core#transformation#maperr,
  - core#transformation#flatmap
position: 0
---

Applies a given project function to each item emitted by the source Observable, and emits the results of these function applications as an Observable sequence.

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5),
    ro.Map(func(i int) string {
        return fmt.Sprintf("Item-%d", i)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Item-1
// Next: Item-2
// Next: Item-3
// Next: Item-4
// Next: Item-5
// Completed
```

### With context

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5),
    ro.MapWithContext(func(ctx context.Context, i int) (context.Context, string) {
        return ctx, fmt.Sprintf("Item-%d", i)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Item-1
// Next: Item-2
// Next: Item-3
// Next: Item-4
// Next: Item-5
// Completed
```

### With index

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5),
    ro.MapI(func(i int, index int64) string {
        return fmt.Sprintf("Item-%d-Index-%d", i, index)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Item-1-Index-0
// Next: Item-2-Index-1
// Next: Item-3-Index-2
// Next: Item-4-Index-3
// Next: Item-5-Index-4
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5),
    ro.MapIWithContext(func(ctx context.Context, i int, index int64) (context.Context, string) {
        return ctx, fmt.Sprintf("Item-%d-Index-%d", i, index)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Item-1-Index-0
// Next: Item-2-Index-1
// Next: Item-3-Index-2
// Next: Item-4-Index-3
// Next: Item-5-Index-4
// Completed
```