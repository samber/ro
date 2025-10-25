---
name: BufferWithTimeOrCount
slug: bufferwithtimeorcount
sourceRef: operator_transformations.go#L396
type: core
category: transformation
signatures:
  - "func BufferWithTimeOrCount[T any](size int, duration time.Duration)"
playUrl: https://go.dev/play/p/NyiF19jUdQD
variantHelpers:
  - core#transformation#bufferwithtimeorcount
similarHelpers:
  - core#transformation#bufferwhen
  - core#transformation#bufferwithcount
  - core#transformation#bufferwithtime
position: 60
---

Buffers the source Observable values until either the buffer reaches the specified size or the specified time duration elapses, whichever occurs first.

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.BufferWithTimeOrCount[int64](5, 300*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Buffers when either condition is met
// Next: [0, 1, 2] (after 300ms - time limit reached)
// Next: [3, 4, 5, 6, 7] (count limit reached)
// Next: [8, 9, 10] (time limit reached)
```

### Time-limited scenario

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.BufferWithTimeOrCount[int64](10, 200*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(800 * time.Millisecond)
sub.Unsubscribe()

// Time limit reached before count
// Next: [0, 1] (after 200ms)
// Next: [2, 3] (after 400ms)
// Next: [4, 5] (after 600ms)
```

### Count-limited scenario

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(50*time.Millisecond), // Fast emissions
    ro.BufferWithTimeOrCount[int64](3, 1000*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Count limit reached before time
// Next: [0, 1, 2] (count reached after 150ms)
// Next: [3, 4, 5] (count reached after 300ms)
// Next: [6, 7, 8] (count reached after 450ms)
```

### Practical example: Batching with safety limits

```go
obs := ro.Pipe[int64, int](
    // Simulate user events
    ro.Interval(30*time.Millisecond),
    ro.Take[int64](20),
    ro.BufferWithTimeOrCount[int64](5, 200*time.Millisecond),
    ro.Map(func(batch []int64) int {
        return len(batch) // Show batch sizes
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Outputs batch sizes based on either count (5) or time (200ms)
// Prevents memory buildup while ensuring timely processing
```