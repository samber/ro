---
name: MergeAll
slug: mergeall
sourceRef: operator_combining.go#L76
type: core
category: combining
signatures:
  - "func MergeAll[T any]()"
playUrl: https://go.dev/play/p/m3nHZZJbwMF
variantHelpers:
  - core#combining#mergeall
similarHelpers:
  - core#combining#merge
  - core#combining#concatall
  - core#combining#combinelatestall
position: 10
---

Creates an Observable that merges items from multiple Observable sources provided as a higher-order Observable, emitting items as they are emitted from any source.

```go
obs := ro.Pipe[Observable[int], int](
    ro.Just(
        ro.Just(1, 2, 3),
        ro.Just(4, 5, 6),
        ro.Just(7, 8, 9),
    ),
    ro.MergeAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 1
// Next: 4
// Next: 7
// Next: 2
// Next: 5
// Next: 8
// Next: 3
// Next: 6
// Next: 9
// Completed
```

### With different emission rates

```go
obs := ro.Pipe[Observable[int64], int64](
    ro.Just(
        ro.Pipe[int64, int64](ro.Interval(100*time.Millisecond), ro.Take[int64](3)),   // Fast: 0,1,2
        ro.Pipe[int64, int64](ro.Interval(200*time.Millisecond), ro.Take[int64](2)),   // Medium: 0,1
        ro.Pipe[int64, int64](ro.Interval(300*time.Millisecond), ro.Take[int64](1)),   // Slow: 0
    ),
    ro.MergeAll[int64](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Values interleaved based on emission timing
// 0, 0, 0, 1, 1, 2
```

### Dynamic observable collection

```go
// Create observables dynamically
observables := []ro.Observable[int]{
    ro.Just(1, 2),
    ro.Just(3, 4),
    ro.Just(5, 6),
}

obs := ro.Pipe[Observable[int], int](
    ro.Just(observables...),
    ro.MergeAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Merges all values from the collection of observables
```

### Edge case: Empty observable of observables

```go
obs := ro.Pipe[Observable[int], int](
    ro.Empty[ro.Observable[int]](),
    ro.MergeAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// No items emitted, completes immediately
// Completed
```

### Edge case: Single observable in collection

```go
obs := ro.Pipe[Observable[int], int](
    ro.Just(ro.Just(1, 2, 3)),
    ro.MergeAll[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```