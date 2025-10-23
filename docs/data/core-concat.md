---
name: Concat
slug: concat
sourceRef: operator_creation.go#L144
type: core
category: combining
signatures:
  - "func Concat[T any](sources ...Observable[T]) Observable[T]"
playUrl:
variantHelpers:
  - core#combining#concat
similarHelpers:
  - core#creation#merge
  - core#combining#concatwith
position: 44
---

Creates an Observable that concatenates multiple source Observables, emitting all items from each source sequentially.

```go
obs := ro.Concat(
    ro.Just(1, 2, 3),
    ro.Just(4, 5, 6),
    ro.Just(7, 8, 9),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3 (from first observable)
// Next: 4, 5, 6 (from second observable)
// Next: 7, 8, 9 (from third observable)
// Completed
```

### With time-based sources

```go
obs := ro.Concat(
    ro.Pipe[int, int](ro.Just(1), ro.Delay(100*time.Millisecond)),
    ro.Pipe[int, int](ro.Just(2), ro.Delay(100*time.Millisecond)),
    ro.Pipe[int, int](ro.Just(3), ro.Delay(100*time.Millisecond)),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Next: 1 (after 100ms)
// Next: 2 (after 200ms total)
// Next: 3 (after 300ms total)
// Completed
```

### With empty observables

```go
obs := ro.Concat(
    ro.Just(1, 2),
    ro.Empty[int](),
    ro.Just(3, 4),
    ro.Empty[int](),
    ro.Just(5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// (empty observable emits nothing)
// Next: 3
// Next: 4
// (empty observable emits nothing)
// Next: 5
// Completed
```

### Error propagation

```go
obs := ro.Concat(
    ro.Just(1, 2),
    ro.Pipe[int, int](ro.Just(3), ro.Throw[int](errors.New("error"))),
    ro.Just(4, 5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2
// Error: error (subsequent observables are not processed)
```
