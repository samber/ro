---
name: Contains
slug: contains
sourceRef: operator_conditional.go#L74
type: core
category: conditional
signatures:
  - "func Contains[T any](predicate func(item T) bool)"
  - "func ContainsWithContext[T any](predicate func(ctx context.Context, item T) bool)"
  - "func ContainsI[T any](predicate func(item T, index int64) bool)"
  - "func ContainsIWithContext[T any](predicate func(ctx context.Context, item T, index int64) bool)"
playUrl: https://go.dev/play/p/ldteqqGsMWM
variantHelpers:
  - core#conditional#contains
  - core#conditional#containswithcontext
  - core#conditional#containsi
  - core#conditional#containsiwithcontext
similarHelpers: []
position: 10
---

Determines whether any element of an observable sequence satisfies a condition.

```go
obs := ro.Pipe[int, bool](
    ro.Just(1, 2, 3, 4, 5),
    ro.Contains(func(i int) bool {
        return i == 3
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
    ro.ContainsWithContext(func(ctx context.Context, n int) bool {
        return n == 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```

### With index

```go
obs := ro.Pipe[string, bool](
    ro.Just("apple", "banana", "cherry"),
    ro.ContainsI(func(item string, index int64) bool {
        return index == 1 && item == "banana"
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
    ro.ContainsIWithContext(func(ctx context.Context, n int, index int64) bool {
        return n > 3 && index >= 3
    }),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Completed
```