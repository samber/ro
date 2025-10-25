---
name: Skip
slug: skip
sourceRef: operator_filter.go#L122
type: core
category: filtering
signatures:
  - "func Skip(count int64)"
playUrl: https://go.dev/play/p/AAEJaUZJuIj
variantHelpers:
  - core#filtering#skip
similarHelpers: []
position: 80
---

Bypasses a specified number of elements in an observable sequence and then returns the remaining elements.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Skip(2),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3
// Next: 4
// Next: 5
// Completed
```

### Skip more than available

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.Skip(5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```

### Skip zero

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Skip(0),
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

### With infinite sequence

```go
obs := ro.Pipe[int64, int64](
    ro.Pipe[time.Time, int64](
        ro.Interval(100*time.Millisecond),
        ro.Take[int64](10),
    ),
    ro.Skip(3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 3 (after 400ms)
// Next: 4 (after 500ms)
// Next: 5 (after 600ms)
// ...
// Next: 9 (after 1000ms)
// Completed
```