---
name: Reduce
slug: reduce
sourceRef: operator_math.go#L310
type: core
category: math
signatures:
  - "func Reduce[T any, R any](accumulator func(agg R, item T) R, seed R)"
  - "func ReduceWithContext[T any, R any](accumulator func(ctx context.Context, agg R, item T) (context.Context, R), seed R)"
  - "func ReduceI[T any, R any](accumulator func(agg R, item T, index int64) R, seed R)"
  - "func ReduceIWithContext[T any, R any](accumulator func(ctx context.Context, agg R, item T, index int64) (context.Context, R), seed R)"
playUrl:
variantHelpers:
  - core#math#reduce
  - core#math#reducewithcontext
  - core#math#reducei
  - core#math#reduceiwithcontext
similarHelpers:
  - core#math#sum
  - core#math#average
  - core#transformation#scan
position: 30
---

Applies an accumulator function over an Observable sequence, and returns the final accumulated result when the source completes.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Reduce(func(acc int, item int) int {
        return acc + item
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 15
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ReduceWithContext(func(ctx context.Context, acc int, item int) (context.Context, int) {
        return ctx, acc + item
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 15
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ReduceI(func(acc int, item int, index int64) int {
        return acc + (item * int(index+1)) // Multiply by position
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 55 (0 + 1*1 + 2*2 + 3*3 + 4*4 + 5*5)
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ReduceIWithContext(func(ctx context.Context, acc int, item int, index int64) (context.Context, int) {
        return ctx, acc + (item * int(index+1))
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 55
// Completed
```

### Reduce to different type

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3, 4, 5),
    ro.Reduce(func(acc string, item int) string {
        return fmt.Sprintf("%s%d", acc, item)
    }, ""),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "12345"
// Completed
```

### Practical example: Building a map

```go
obs := ro.Pipe[string, map[string]int](
    ro.Just("apple", "banana", "cherry"),
    ro.Reduce(func(acc map[string]int, item string) map[string]int {
        acc[item] = len(item)
        return acc
    }, make(map[string]int)),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]int]())
defer sub.Unsubscribe()

// Next: map[apple:5 banana:6 cherry:6]
// Completed
```

### Reduce with no seed (first item as seed)

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Reduce(func(acc int, item int) int {
        return acc * item
    }, 1), // 1 is the seed
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 120 (1 * 2 * 3 * 4 * 5)
// Completed
```