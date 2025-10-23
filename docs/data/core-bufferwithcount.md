---
name: BufferWithCount
slug: bufferwithcount
sourceRef: operator_transformations.go#L426
type: core
category: transformation
signatures:
  - "func BufferWithCount[T any](size int)"
playUrl:
variantHelpers:
  - core#transformation#bufferwithcount
similarHelpers:
  - core#transformation#bufferwhen
  - core#transformation#bufferwithtime
  - core#transformation#bufferwithtimeorcount
position: 40
---

Buffers the source Observable values into non-overlapping buffers of a specific size, and emits these buffers as arrays.

```go
obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.BufferWithCount[int](3),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1, 2, 3]
// Next: [4, 5, 6]
// Next: [7, 8, 9]
// Next: [10]
// Completed
```

### Practical example: Batch processing

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.BufferWithCount[int](4),
    ro.Map(func(batch []int) int {
        // Process batch of items
        sum := 0
        for _, item := range batch {
            sum += item
        }
        return sum
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 10 (1+2+3+4)
// Next: 22 (5+6+7+8)
// Next: 19 (9+10)
// Completed
```

### Edge case: Single item buffer

```go
obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3, 4, 5),
    ro.BufferWithCount[int](1),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1]
// Next: [2]
// Next: [3]
// Next: [4]
// Next: [5]
// Completed
```

### Edge case: Buffer larger than source

```go
obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3),
    ro.BufferWithCount[int](10),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1, 2, 3]
// Completed
```