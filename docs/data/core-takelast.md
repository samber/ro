---
name: TakeLast
slug: takelast
sourceRef: operator_filter.go#L414
type: core
category: filtering
signatures:
  - "func TakeLast(count int64)"
playUrl:
variantHelpers:
  - core#filtering#takelast
similarHelpers: []
position: 260
---

Returns a specified number of contiguous elements from the end of an observable sequence.

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5),
    ro.TakeLast(2),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 4
// Next: 5
// Completed
```

### Take all elements

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3),
    ro.TakeLast(5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### TakeLast with zero

```go
obs := ro.Pipe(
    ro.Just(1, 2, 3, 4, 5),
    ro.TakeLast(0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Completed (no values emitted)
```

### With strings

```go
obs := ro.Pipe(
    ro.Just("apple", "banana", "cherry", "date", "elderberry"),
    ro.TakeLast(3),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: cherry
// Next: date
// Next: elderberry
// Completed
```

### Memory usage note

```go
// TakeLast must buffer all elements to determine the last N
// Use with caution on very long or infinite sequences
obs := ro.Pipe(
    ro.Pipe(
        ro.Interval(100*time.Millisecond),
        ro.Take[int64](10),
    ),
    ro.TakeLast(3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 7
// Next: 8
// Next: 9
// Completed
```