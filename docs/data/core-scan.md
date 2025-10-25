---
name: Scan
slug: scan
sourceRef: operator_transformations.go#L245
type: core
category: transformation
signatures:
  - "func Scan[T any, R any](reduce func(accumulator R, item T) R, seed R)"
  - "func ScanWithContext[T any, R any](reduce func(ctx context.Context, accumulator R, item T) (context.Context, R), seed R)"
  - "func ScanI[T any, R any](reduce func(accumulator R, item T, index int64) R, seed R)"
  - "func ScanIWithContext[T any, R any](reduce func(ctx context.Context, accumulator R, item T, index int64) (context.Context, R), seed R)"
playUrl: https://go.dev/play/p/gAzVq-a0Jiz
variantHelpers:
  - core#transformation#scan
  - core#transformation#scanwithcontext
  - core#transformation#scani
  - core#transformation#scaniwithcontext
similarHelpers:
  - core#transformation#reduce
  - core#math#sum
position: 20
---

Applies an accumulator function over an Observable sequence, and returns each intermediate result, with an optional seed value.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Scan(func(acc int, item int) int {
        return acc + item
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 3
// Next: 6
// Next: 10
// Next: 15
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ScanWithContext(func(ctx context.Context, acc int, item int) (context.Context, int) {
        return ctx, acc + item
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 3
// Next: 6
// Next: 10
// Next: 15
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ScanI(func(acc int, item int, index int64) int {
        return acc + (item * int(index+1)) // Multiply by position
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1 (0 + 1*1)
// Next: 5 (1 + 2*2)
// Next: 14 (5 + 3*3)
// Next: 30 (14 + 4*4)
// Next: 55 (30 + 5*5)
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.ScanIWithContext(func(ctx context.Context, acc int, item int, index int64) (context.Context, int) {
        return ctx, acc + (item * int(index+1))
    }, 0),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1 (0 + 1*1)
// Next: 5 (1 + 2*2)
// Next: 14 (5 + 3*3)
// Next: 30 (14 + 4*4)
// Next: 55 (30 + 5*5)
// Completed
```

### Practical example: Building a string

```go
obs := ro.Pipe[string, string](
    ro.Just("hello", "world", "rx"),
    Scan(func(acc string, item string) string {
        if acc == "" {
            return item
        }
        return acc + " " + item
    }, ""),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: "hello"
// Next: "hello world"
// Next: "hello world rx"
// Completed
```