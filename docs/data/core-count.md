---
name: Count
slug: count
sourceRef: operator_math.go#L60
type: core
category: math
signatures:
  - "func Count[T any]()"
playUrl:
variantHelpers:
  - core#math#count
similarHelpers:
  - core#math#sum
  - core#math#average
  - core#math#reduce
position: 10
---

Counts the number of items emitted by an Observable and emits the total count when the source completes.

```go
obs := ro.Pipe[int, int64](
    ro.Just(1, 2, 3, 4, 5),
    ro.Count[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 5
// Completed
```

### Count with empty observable

```go
obs := ro.Pipe[int, int64](
    ro.Empty[int](),
    ro.Count[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 0
// Completed
```

### Count with single value

```go
obs := ro.Pipe[string, int64](
    ro.Just("hello"),
    ro.Count[string](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 1
// Completed
```

### Count with complex types

```go
type Person struct {
    Name string
    Age  int
}

obs := ro.Pipe[Person, int64](
    ro.Just(
        Person{"Alice", 25},
        Person{"Bob", 30},
        Person{"Charlie", 35},
    ),
    ro.Count[Person](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 3
// Completed
```

### Count after filtering

```go
obs := ro.Pipe[int, int64](
    ro.Just(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    ro.Filter(func(i int) bool {
        return i%2 == 0 // Count even numbers
    }),
    ro.Count[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 5
// Completed
```