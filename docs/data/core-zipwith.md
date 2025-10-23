---
name: ZipWith
slug: zipwith
sourceRef: operator_creation.go#L479
type: core
category: combining
signatures:
  - "func ZipWith[A any, B any](obsB Observable[B])"
  - "func ZipWith1[A any, B any](obsB Observable[B])"
  - "func ZipWith2[A any, B any, C any](obsB Observable[B], obsC Observable[C])"
  - "func ZipWith3[A any, B any, C any, D any](obsB Observable[B], obsC Observable[C], obsD Observable[D])"
  - "func ZipWith4[A any, B any, C any, D any, E any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E])"
  - "func ZipWith5[A any, B any, C any, D any, E any, F any](obsB Observable[B], obsC Observable[C], obsD Observable[D], obsE Observable[E], obsF Observable[F])"
playUrl:
variantHelpers:
  - core#combining#zipwith
  - core#combining#zipwith1
  - core#combining#zipwith2
  - core#combining#zipwith3
  - core#combining#zipwith4
  - core#combining#zipwith5
similarHelpers:
  - core#combining#zip
  - core#combining#zipall
  - core#combining#combinelatestwith
position: 21
---

Creates an Observable that combines values from the source Observable with other source Observables using a pipe operator pattern, emitting tuples of zipped values.

### ZipWith

```go
obs := ro.Pipe[int, lo.Tuple2[int, string]](
    ro.Just(1, 2, 3),
    ro.ZipWith(ro.Just("A", "B", "C")),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[int, string]]())
defer sub.Unsubscribe()

// Next: (1, A)
// Next: (2, B)
// Next: (3, C)
// Completed
```

### ZipWith2 (three sources)

```go
obs := ro.Pipe[int, lo.Tuple3[int, string, bool]](
    ro.Just(1, 2, 3),
    ro.ZipWith2(
        ro.Just("A", "B", "C"),
        ro.Just(true, false, true),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple3[int, string, bool]]())
defer sub.Unsubscribe()

// Next: (1, A, true)
// Next: (2, B, false)
// Next: (3, C, true)
// Completed
```

### ZipWith3 (four sources)

```go
obs := ro.Pipe[int, lo.Tuple4[int, string, bool, float64]](
    ro.Just(1, 2),
    ro.ZipWith3(
        ro.Just("A", "B"),
        ro.Just(true, false),
        ro.Just(1.1, 2.2),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple4[int, string, bool, float64]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1)
// Next: (2, B, false, 2.2)
// Completed
```

### ZipWith4 (five sources)

```go
obs := ro.Pipe[int, lo.Tuple5[int, string, bool, float64, []int]](
    ro.Just(1, 2),
    ro.ZipWith4(
        ro.Just("A", "B"),
        ro.Just(true, false),
        ro.Just(1.1, 2.2),
        ro.Just([]int{10, 20}),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple5[int, string, bool, float64, []int]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1, [10, 20])
// Next: (2, B, false, 2.2, [10, 20])
// Completed
```

### ZipWith5 (six sources)

```go
obs := ro.Pipe[int, lo.Tuple6[int, string, bool, float64, []int, string]](
    ro.Just(1, 2),
    ro.ZipWith5(
        ro.Just("A", "B"),
        ro.Just(true, false),
        ro.Just(1.1, 2.2),
        ro.Just([]int{10, 20}),
        ro.Just("x", "y"),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple6[int, string, bool, float64, []int, string]]())
defer sub.Unsubscribe()

// Next: (1, A, true, 1.1, [10, 20], "x")
// Next: (2, B, false, 2.2, [10, 20], "y")
// Completed
```

### With different emission rates

```go
obs := ro.Pipe[int, lo.Tuple2[int, string]](
    ro.Just(1, 2, 3),
    ro.ZipWith(
        ro.Pipe[string, string](ro.Interval(100*time.Millisecond), ro.Take[string](3)), // "A", "B", "C"
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[int, string]]())
time.Sleep(500 * time.Millisecond)
sub.Unsubscribe()

// Values are zipped as they arrive
// Next: (1, A)
// Next: (2, B)
// Next: (3, C)
// Completed
```

### Edge case: Different length observables

```go
obs := ro.Pipe[int, lo.Tuple3[int, string, bool]](
    ro.Just(1, 2, 3, 4, 5), // 5 items
    ro.ZipWith2(
        ro.Just("A", "B"),     // 2 items
        ro.Just(true, false),  // 2 items
    ),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple3[int, string, bool]]())
defer sub.Unsubscribe()

// Next: (1, A, true)
// Next: (2, B, false)
// Completed
// Only zips up to the shortest observable
```