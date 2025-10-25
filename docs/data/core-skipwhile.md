---
name: SkipWhile
slug: skipwhile
sourceRef: operator_filter.go#L155
type: core
category: filtering
signatures:
  - "func SkipWhile[T any](predicate func(item T) bool)"
  - "func SkipWhileWithContext[T any](predicate func(ctx context.Context, item T) bool)"
  - "func SkipWhileI[T any](predicate func(item T, index int64) bool)"
  - "func SkipWhileIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool)"
playUrl: https://go.dev/play/p/Mb1cyMSD0Sc
variantHelpers:
  - core#filtering#skipwhile
  - core#filtering#skipwhilewithcontext
  - core#filtering#skipwhilei
  - core#filtering#skipwhileiwithcontext
similarHelpers: []
position: 90
---

Bypasses elements in an observable sequence as long as a specified condition is true, and then returns the remaining elements.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.SkipWhile(func(i int) bool {
        return i < 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Next: 4
// Next: 5
// Completed
```

### With context

```go
obs := ro.Pipe[string, string](
    ro.Just("apple", "banana", "cherry", "date"),
    ro.SkipWhileWithContext(func(ctx context.Context, fruit string) bool {
        return strings.HasPrefix(fruit, "a") || strings.HasPrefix(fruit, "b")
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: cherry
// Next: date
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(10, 20, 30, 40, 50),
    ro.SkipWhileI(func(item int, index int64) bool {
        return index < 2 // Skip first 2 items regardless of value
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 30
// Next: 40
// Next: 50
// Completed
```

### With index and context

```go
obs := ro.Pipe[string, string](
    ro.Just("a", "b", "c", "d", "e"),
    ro.SkipWhileIWithContext(func(ctx context.Context, item string, index int64) bool {
        return index < 3 && item != "d"
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: d
// Next: e
// Completed
```

### When condition never becomes false

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.SkipWhile(func(i int) bool {
        return i > 0 // Always true
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```