---
name: Merge
slug: merge
sourceRef: operator_combining.go#L27
type: core
category: combining
signatures:
  - "func Merge[T any](sources ...Observable[T])"
playUrl: https://go.dev/play/p/Nerpzkth1lT
variantHelpers:
  - core#combining#merge
similarHelpers:
  - core#combining#concat
  - core#combining#mergewith
  - core#combining#mergeall
  - core#combining#zipwith
position: 0
---

Creates an Observable that emits items from multiple source Observables, interleaved as they are emitted.

### Merge multiple sources

```go
obs := ro.Merge(
    ro.Just(1, 2, 3),
    ro.Just(4, 5, 6),
    ro.Just(7, 8, 9),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Order may vary due to interleaving
// Next: 1
// Next: 4
// Next: 7
// Next: 2
// Next: 5
// Next: 8
// Next: 3
// Next: 6
// Next: 9
// Completed
```

### With different emission rates

```go
obs := ro.Merge(
    ro.Pipe[time.Time, int64](ro.Interval(100*time.Millisecond), ro.Take[int64](3)),   // 0,1,2
    ro.Pipe[time.Time, int64](ro.Interval(200*time.Millisecond), ro.Take[int64](2)),   // 0,1
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(800 * time.Millisecond)
sub.Unsubscribe()

// Values interleaved based on emission timing
// 0, 0, 1, 1, 2
```

### With hot observables

```go
source1 := ro.Interval(100 * time.Millisecond)
source2 := ro.Interval(150 * time.Millisecond)
obs := ro.Merge(
    ro.Pipe[time.Time, int64](source1, ro.Take[int64](5)),
    ro.Pipe[time.Time, int64](source2, ro.Take[int64](3)),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Values interleaved based on actual emission times
```

### With error handling

```go
obs := ro.Merge(
    ro.Just(1, 2, 3),
    ro.Pipe[int, int](
        ro.Just(4, 5, 6),
        ro.MapErr(func(i int) (int, error) {
            if i == 5 {
                return 0, fmt.Errorf("error on 5")
            }
            return i, nil
        }),
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
        fmt.Println("Complete")
    },
))
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

// Error terminates the entire merged observable
```

### With single source

```go
obs := ro.Merge(
    ro.Just(1, 2, 3),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```

### With no sources

```go
obs := ro.Merge[int]()

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Immediately completes with no values
```