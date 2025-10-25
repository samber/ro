---
name: Zip
slug: zip
sourceRef: operator_creation.go#L473
type: core
category: combining
signatures:
  - "func Zip[T any](sources ...Observable[T])"
  - "func Zip2[A any, B any](obsA Observable[A], obsB Observable[B])"
  - "func Zip3[A any, B any, C any](obsA Observable[A], obsB Observable[B], obsC Observable[C])"
  - "func Zip4[A any, B any, C any, D any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D])"
  - "func Zip5[A any, B any, C any, D any, E any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E])"
  - "func Zip6[A any, B any, C any, D any, E any, F any](obsA Observable[A], obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E], obsF Observable[F])"
playUrl: https://go.dev/play/p/5YxbQ5jNzjQ
variantHelpers:
  - core#combining#zipx
  - core#combining#zip
  - core#combining#zip3
  - core#combining#zip4
  - core#combining#zip5
  - core#combining#zip6
similarHelpers:
  - core#combining#zipwith
  - core#combining#zipall
  - core#combining#combinelatestx
position: 20
---

Creates an Observable that combines the values from multiple source Observables by emitting tuples or arrays of values in the order they were zipped.

### Zip2

```go
obs := ro.Zip2(
    ro.Just(1, 2, 3),
    ro.Just("A", "B", "C"),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[int, string]]())
defer sub.Unsubscribe()

// Next: (1, A)
// Next: (2, B)
// Next: (3, C)
// Completed
```

### Zip3

```go
obs := ro.Zip3(
    ro.Just(1, 2, 3),
    ro.Just("A", "B", "C"),
    ro.Just(true, false, true),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple3[int, string, bool]]())
defer sub.Unsubscribe()

// Next: (1, A, true)
// Next: (2, B, false)
// Next: (3, C, true)
// Completed
```

### Zip4

```go
obs := ro.Zip4(
    ro.Just(1, 2),
    ro.Just("A", "B"),
    ro.Just(true, false),
    ro.Just(1.1, 2.2),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple4[int, string, bool, float64]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1)
// Next: (2, B, false, 2.2)
// Completed
```

### Zip5

```go
obs := ro.Zip5(
    ro.Just(1, 2),
    ro.Just("A", "B"),
    ro.Just(true, false),
    ro.Just(1.1, 2.2),
    ro.Just([]int{10, 20}),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple5[int, string, bool, float64, []int]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1, [10, 20])
// Next: (2, B, false, 2.2, [10, 20])
// Completed
```

### Zip6

```go
obs := ro.Zip6(
    ro.Just(1, 2),
    ro.Just("A", "B"),
    ro.Just(true, false),
    ro.Just(1.1, 2.2),
    ro.Just([]int{10, 20}),
    ro.Just("x", "y"),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple6[int, string, bool, float64, []int, string]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1, [10, 20], "x")
// Next: (2, B, false, 2.2, [10, 20], "y")
// Completed
```

### Zip with multiple sources

```go
obs := ro.Zip(
    ro.Just(1, 2, 3),
    ro.Just("A", "B", "C"),
    ro.Just(true, false, true),
)

sub := obs.Subscribe(ro.PrintObserver[[]any]())
defer sub.Unsubscribe()

// Next: [1, A, true]
// Next: [2, B, false]
// Next: [3, C, true]
// Completed
```

### Edge case: Different length observables

```go
obs := ro.Zip2(
    ro.Just(1, 2, 3, 4, 5), // 5 items
    ro.Just("A", "B"),     // 2 items
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[int, string]]())
defer sub.Unsubscribe()

// Next: (1, A)
// Next: (2, B)
// Completed
// Only zips up to the shortest observable
```

### Edge case: Single observable

```go
obs := ro.Zip(
    ro.Just(1, 2, 3),
)

sub := obs.Subscribe(ro.PrintObserver[[]any]())
defer sub.Unsubscribe()

// Next: [1]
// Next: [2]
// Next: [3]
// Completed
```