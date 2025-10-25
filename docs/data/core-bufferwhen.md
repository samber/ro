---
name: BufferWhen
slug: bufferwhen
sourceRef: operator_transformations.go#L376
type: core
category: transformation
signatures:
  - "func BufferWhen[T any, B any](boundary Observable[B])"
playUrl: https://go.dev/play/p/w8c_zuaLl9l
variantHelpers:
  - core#transformation#bufferwhen
similarHelpers:
  - core#transformation#bufferwithcount
  - core#transformation#bufferwithtime
  - core#transformation#bufferwithtimeorcount
position: 30
---

Buffers the source Observable values until a boundary Observable emits an item, then emits the buffered values as an array.

```go
// Create boundary observable that emits every 3 items
boundary := ro.Pipe[int64, int64](
    ro.Interval(200*time.Millisecond),
    ro.Take[int64](3),
)

obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.BufferWhen[int64, int64](boundary),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Buffers when boundary observable emits
// Next: [0, 1] (after first boundary)
// Next: [2, 3, 4] (after second boundary)
// Next: [5, 6, 7] (after third boundary)
```

### With custom boundary

```go
// Create boundary based on clicks or events
clickBoundary := ro.Pipe[int64, int64](
    ro.Interval(500*time.Millisecond),
    ro.Take[int64](2),
)

obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8),
    ro.BufferWhen[int, int64](clickBoundary),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1, 2, 3, 4] (after first boundary at 500ms)
// Next: [5, 6, 7, 8] (after second boundary at 1000ms)
// Completed
```

### Edge case: Empty source

```go
boundary := ro.Just("trigger")
obs := ro.Pipe[int, []int](
    ro.Empty[int](),
    ro.BufferWhen[int, string](boundary),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [] (empty buffer when boundary emits)
// Completed
```