---
name: Pairwise
slug: pairwise
sourceRef: operator_combining.go#L945
type: core
category: combining
signatures:
  - "func Pairwise[T any]()"
playUrl:
variantHelpers:
  - core#combining#pairwise
similarHelpers:
  - core#combining#scan
  - core#combining#zip
  - core#combining#bufferwithcount
position: 80
---

Emits the previous and current values as a pair (array of two values). The first value doesn't emit until the second value arrives.

```go
obs := ro.Pipe[string, []string](
    ro.Just("a", "b", "c", "d"),
    ro.Pairwise(),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Next: [a b]
// Next: [b c]
// Next: [c d]
// Completed
```

### With numbers

```go
obs := ro.Pipe[int, []int](
    ro.Just(1, 2, 3, 4, 5),
    ro.Pairwise(),
)

sub := obs.Subscribe(ro.PrintObserver[[]int]())
defer sub.Unsubscribe()

// Next: [1 2]
// Next: [2 3]
// Next: [3 4]
// Next: [4 5]
// Completed
```

### With single value

```go
obs := ro.Pipe[string, []string](
    ro.Just("only one"),
    ro.Pairwise(),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Completed (no pairs emitted)
```

### With empty observable

```go
obs := ro.Pipe[string, []string](
    ro.Empty[string](),
    ro.Pairwise(),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Completed (no pairs emitted)
```

### With time-based emissions

```go
obs := ro.Pipe[int64, []int64](
    ro.Interval(100*time.Millisecond),
    ro.Pairwise(),
    ro.Take(4),
)

sub := obs.Subscribe(ro.PrintObserver[[]int64]())
time.Sleep(600 * time.Millisecond)
sub.Unsubscribe()

// Next: [0 1]
// Next: [1 2]
// Next: [2 3]
// Next: [3 4]
// Completed
```

### With custom types

```go
type Point struct {
    X, Y int
}

obs := ro.Pipe[Point, []Point](
    ro.Just(
        Point{1, 2},
        Point{3, 4},
        Point{5, 6},
    ),
    ro.Pairwise(),
)

sub := obs.Subscribe(ro.PrintObserver[[]Point]())
defer sub.Unsubscribe()

// Next: [{1 2} {3 4}]
// Next: [{3 4} {5 6}]
// Completed
```