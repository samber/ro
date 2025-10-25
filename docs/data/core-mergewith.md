---
name: MergeWith
slug: mergewith
sourceRef: operator_creation.go#L451
type: core
category: combining
signatures:
  - "func MergeWith[T any](obsB ro.Observable[T])"
  - "func MergeWith1[T any](obsB ro.Observable[T])"
  - "func MergeWith2[T any](obsB ro.Observable[T], obsC ro.Observable[T])"
  - "func MergeWith3[T any](obsB ro.Observable[T], obsC ro.Observable[T], obsD ro.Observable[T])"
  - "func MergeWith4[T any](obsB ro.Observable[T], obsC ro.Observable[T], obsD ro.Observable[T], obsE ro.Observable[T])"
  - "func MergeWith5[T any](obsB ro.Observable[T], obsC ro.Observable[T], obsD ro.Observable[T], obsE ro.Observable[T], obsF ro.Observable[T])"
playUrl: https://go.dev/play/p/6QpUzcdRWJl
variantHelpers:
  - core#combining#mergewith
  - core#combining#mergewith1
  - core#combining#mergewith2
  - core#combining#mergewith3
  - core#combining#mergewith4
  - core#combining#mergewith5
similarHelpers:
  - core#combining#merge
  - core#combining#concat
  - core#combining#combineLatestWith
position: 10
---

Creates an Observable that merges emissions from the source Observable with additional source Observables, interleaved as they are emitted.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeWith(ro.Just(4, 5, 6)),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 1
// Next: 4
// Next: 2
// Next: 5
// Next: 3
// Next: 6
// Completed
```

### MergeWith1 (alias for MergeWith)

```go
obs := ro.Pipe[string, string](
    ro.Just("A", "B", "C"),
    ro.MergeWith1(ro.Just("D", "E", "F")),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// MergeWith1 is an alias for MergeWith
// Values interleaved from both sources
```

### MergeWith2 (three sources)

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeWith2(
        ro.Just(4, 5, 6),
        ro.Just(7, 8, 9),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Combines source with two additional observables
// Next values will be interleaved from all three sources
```

### MergeWith3 (four sources)

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.MergeWith3(
        ro.Just(3, 4),
        ro.Just(5, 6),
        ro.Just(7, 8),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Combines source with three additional observables
// All values interleaved as they're emitted
```

### MergeWith4 (five sources)

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.MergeWith4(
        ro.Just(3, 4),
        ro.Just(5, 6),
        ro.Just(7, 8),
        ro.Just(9, 10),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Combines source with four additional observables
```

### MergeWith5 (six sources)

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.MergeWith5(
        ro.Just(3, 4),
        ro.Just(5, 6),
        ro.Just(7, 8),
        ro.Just(9, 10),
        ro.Just(11, 12),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Combines source with five additional observables
// Maximum convenience method for merging six sources
```

### With hot observables

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100 * time.Millisecond),
    ro.Take[int64](3),
    ro.MergeWith2(
        ro.Pipe[int64, int64](ro.Interval(150 * time.Millisecond), ro.Take[int64](2)),
        ro.Pipe[int64, int64](ro.Interval(200 * time.Millisecond), ro.Take[int64](2)),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Values interleaved based on actual emission timing from all sources
```

### With error propagation

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeWith(
        ro.Pipe[int, int](
            ro.Just(4, 5, 6),
            ro.MapErr(func(i int) (int, error) {
                if i == 5 {
                    return 0, fmt.Errorf("error on 5")
                }
                return i, nil
            }),
        ),
    ),
)

sub := obs.Subscribe(ro.NewObserver(
    func(value int) {
        fmt.Printf("Next: %d\n", value)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    }
))
time.Sleep(100 * time.Millisecond)
defer sub.Unsubscribe()

// Error from any source terminates the entire merged observable
```

### With different data types

```go
obs := ro.Pipe[any, any](
    ro.Just("hello", "world"),
    ro.MergeWith2(
        ro.Just(1, 2, 3),
        ro.Just(true, false),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[any]())
defer sub.Unsubscribe()

// All values can be merged as any type
// Values interleaved as they're emitted from each source
```

### With async operations

```go
obs := ro.Pipe[string, string](
    ro.Just("task1", "task2"),
    ro.MergeWith(
        ro.Pipe[string, string](
            ro.Just("async1", "async2"),
            ro.MapAsync(func(task string) ro.Observable[string] {
                return ro.Defer(func() ro.Observable[string] {
                    time.Sleep(50 * time.Millisecond)
                    return ro.Just(task + "_done")
                })
            }, 2),
        ),
    ),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(300 * time.Millisecond)
defer sub.Unsubscribe()

// Values interleaved as async operations complete
```

### With conditional merging

```go
shouldMerge := true
var secondary ro.Observable[int] = ro.Just(4, 5, 6)

if !shouldMerge {
    secondary = ro.Empty[int]()
}

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.MergeWith(secondary),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Conditional merging based on runtime logic
```