---
name: Repeat
slug: repeat
sourceRef: operator_creation.go#L70
type: core
category: creation
signatures:
  - "func Repeat[T any](source Observable[T])"
playUrl: https://go.dev/play/p/CUvh_TYALNe
variantHelpers:
  - core#creation#repeat
similarHelpers:
  - core#creation#repeatwithinterval
position: 37
---

Creates an Observable that repeats the source Observable sequence when it completes.

```go
source := ro.Just(1, 2, 3)
obs := ro.Repeat(source)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

// Will repeat indefinitely:
// Next: 1, 2, 3, 1, 2, 3, 1, 2, 3, ...
```

### With Take for limited repetitions

```go
source := ro.Just(1, 2, 3)
obs := ro.Pipe[int, int](
    ro.Repeat(source),
    ro.Take[int](8),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1, 2, 3, 1, 2, 3, 1, 2
// Completed (after taking 8 items)
```

### Repeat with interval

```go
source := ro.Pipe[string, string](
    ro.Just("tick"),
    ro.RepeatWithInterval(source, 1*time.Second),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(3500 * time.Millisecond)
sub.Unsubscribe()

// Next: "tick" (immediately)
// Next: "tick" (after 1 second)
// Next: "tick" (after 2 seconds)
// Next: "tick" (after 3 seconds)
```

### Repeat complex sequence

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(100*time.Millisecond),
    ro.Take[int64](3), // 0, 1, 2
    ro.Repeat(source),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
time.Sleep(1000 * time.Millisecond)
sub.Unsubscribe()

// Will repeat the sequence 0, 1, 2 indefinitely
// 0, 1, 2, 0, 1, 2, 0, 1, 2, ...
```

### Repeat with error handling

```go
source := ro.Just(1, 2)
obs := ro.Pipe[int, int](
    ro.Repeat(source),
    ro.Take[int](5),
    ro.Map(func(i int) int {
        if i == 3 {
            panic("error at 3")
        }
        return i
    }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(100 * time.Millisecond)
sub.Unsubscribe()

// Next: 1, 2, 1, 2
// Error: error at 3
```