---
name: BufferWithTime
slug: bufferwithtime
sourceRef: operator_transformations.go#L442
type: core
category: transformation
signatures:
  - "func BufferWithTime[T any](duration time.Duration)"
playUrl:
variantHelpers:
  - core#transformation#bufferwithtime
similarHelpers:
  - core#transformation#bufferwhen
  - core#transformation#bufferwithcount
  - core#transformation#bufferwithtimeorcount
position: 50
---

Buffers the source Observable values for a specified time duration, then emits the buffered values as an array.

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.BufferWithTime[int64](300*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Next: [0, 1, 2] (after 300ms)
// Next: [3, 4, 5] (after 600ms)
// Next: [6, 7, 8] (after 900ms)
```

### With sparse emissions

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.Take[int64](10),
    // Add delay to make emissions sparse
    ro.Map(func(i int64) int64 {
        time.Sleep(50 * time.Millisecond)
        return i
    }),
    ro.BufferWithTime[int64](250*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(2000 * time.Millisecond)
sub.Unsubscribe()

// Buffers based on time, not item count
// Next: [0] (after 250ms)
// Next: [1, 2] (after 500ms)
// Next: [3, 4] (after 750ms)
// Next: [5] (after 1000ms)
```

### Practical example: Debounced batching

```go
obs := ro.Pipe[string, []string](
    ro.Just("A", "B", "C", "D", "E"),
    // Simulate rapid events
    ro.Map(func(s string) string {
        time.Sleep(10 * time.Millisecond)
        return s
    }),
    ro.BufferWithTime[string](100*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Depending on timing, might get:
// Next: ["A", "B", "C", "D", "E"] (all within 100ms)
// Or split into smaller batches based on timing
```