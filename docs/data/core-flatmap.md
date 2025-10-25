---
name: FlatMap
slug: flatmap
sourceRef: operator_transformations.go#L149
type: core
category: transformation
signatures:
  - "func FlatMap[T any, R any](project func(item T) Observable[R])"
  - "func FlatMapWithContext[T any, R any](project func(ctx context.Context, item T) Observable[R])"
  - "func FlatMapI[T any, R any](project func(item T, index int64) Observable[R])"
  - "func FlatMapIWithContext[T any, R any](project func(ctx context.Context, item T, index int64) Observable[R])"
playUrl: https://go.dev/play/p/QBkDMwskibT
variantHelpers:
  - core#transformation#flatmap
  - core#transformation#flatmapwithcontext
  - core#transformation#flatmapi
  - core#transformation#flatmapiwithcontext
similarHelpers:
  - core#transformation#map
  - core#combining#mergemap
position: 10
---

Applies a given project function to each item emitted by the source Observable, where the project function returns an Observable, and then flattens the resulting Observables into a single Observable.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.FlatMap(func(i int) Observable[int] {
        return ro.Just(i*10, i*10+1, i*10+2)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 10
// Next: 20
// Next: 11
// Next: 30
// Next: 21
// Next: 31
// Next: 12
// Next: 22
// Next: 32
// Completed
```

### With context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.FlatMapWithContext(func(ctx context.Context, i int) Observable[int] {
        return ro.Just(i*10, i*10+1, i*10+2)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 10
// Next: 20
// Next: 11
// Next: 30
// Next: 21
// Next: 31
// Next: 12
// Next: 22
// Next: 32
// Completed
```

### With index

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.FlatMapI(func(i int, index int64) Observable[int] {
        return ro.Just(i*10+int(index))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 10
// Next: 20
// Next: 11
// Next: 30
// Next: 21
// Next: 31
// Completed
```

### With index and context

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.FlatMapIWithContext(func(ctx context.Context, i int, index int64) Observable[int] {
        return ro.Just(i*10+int(index))
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 10
// Next: 20
// Next: 11
// Next: 30
// Next: 21
// Next: 31
// Completed
```

### Practical example: Converting single items to multiple

```go
obs := ro.Pipe[string, rune](
    ro.Just("hello", "world"),
    ro.FlatMap(func(s string) Observable[rune] {
        runes := []rune(s)
        return ro.Just(runes...)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[rune]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 'h'
// Next: 'e'
// Next: 'l'
// Next: 'l'
// Next: 'o'
// Next: 'w'
// Next: 'o'
// Next: 'r'
// Next: 'l'
// Next: 'd'
// Completed
```