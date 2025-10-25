---
name: MergeMap
slug: mergemap
sourceRef: operator_combining.go#L108
type: core
category: combining
signatures:
  - "func MergeMap[T any, R any](project func(value T) Observable[R])"
  - "func MergeMapI[T any, R any](project func(value T, index int64) Observable[R])"
  - "func MergeMapWithContext[T any, R any](project func(ctx context.Context, value T) (context.Context, Observable[R]))"
  - "func MergeMapIWithContext[T any, R any](project func(ctx context.Context, value T, index int64) (context.Context, Observable[R]))"
playUrl: https://go.dev/play/p/NwEyrLITshG
variantHelpers:
  - core#combining#mergemap
  - core#combining#mergemapi
  - core#combining#mergemapwithcontext
  - core#combining#mergemapiwithcontext
similarHelpers:
  - core#combining#merge
  - core#combining#mergeall
  - core#combining#flatmap
  - core#combining#switchmap
position: 30
---

Transforms each item from the source Observable into an Observable, then merges the resulting Observables, emitting all items as they are emitted from any transformed Observable.

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3),
    ro.MergeMap(func(n int) Observable[string] {
        return ro.Just(fmt.Sprintf("item-%d", n))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: item-1
// Next: item-2
// Next: item-3
// Completed
```

### With index parameter

```go
obs := ro.Pipe[string, string](
    ro.Just("A", "B", "C"),
    ro.MergeMapI(func(letter string, index int64) Observable[string] {
        return ro.Just(fmt.Sprintf("%s-%d", letter, index))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: A-0
// Next: B-1
// Next: C-2
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeMapWithContext(func(ctx context.Context, n int) (context.Context, Observable[int]) {
        // Can use context for cancellation or values
        return ctx, ro.Just(n * 10)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 10
// Next: 20
// Next: 30
// Completed
```

### With both context and index

```go
obs := ro.Pipe[string, int](
    ro.Just("hello", "world"),
    ro.MergeMapIWithContext(func(ctx context.Context, word string, index int64) (context.Context, Observable[int]) {
        return ctx, ro.Just(len(word))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 5
// Next: 5
// Completed
```

### With async operations

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeMap(func(n int) Observable[int] {
        return ro.Pipe[int64, int](
            ro.Interval(100*time.Millisecond),
            ro.Take[int64](2),
            ro.Map(func(_ int64) int { return n }),
        )
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Values interleaved from all inner observables
// Example output: 1, 1, 2, 2, 3, 3
```

### With error handling

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeMap(func(n int) Observable[int] {
        if n == 2 {
            return ro.Error[int](errors.New("error for 2"))
        }
        return ro.Just(n * 10)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 10
// Next: 20
// Next: 30
// Completed
```

### Edge case: Empty source

```go
obs := ro.Pipe[int, string](
    ro.Empty[int](),
    ro.MergeMap(func(n int) Observable[string] {
        return ro.Just(fmt.Sprintf("item-%d", n))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// No items emitted, completes immediately
// Completed
```