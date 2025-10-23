---
name: RepeatWithInterval
slug: repeatwithinterval
sourceRef: operator_creation.go#L74
type: core
category: creation
signatures:
  - "func RepeatWithInterval[T any](source Observable[T], interval time.Duration)"
playUrl:
variantHelpers:
  - core#creation#repeatwithinterval
similarHelpers:
  - core#creation#repeat
position: 38
---

Creates an Observable that repeats the source Observable sequence after a specified interval when it completes.

```go
source := ro.Just("tick")
obs := ro.RepeatWithInterval(source, 1*time.Second)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(3500 * time.Millisecond)
sub.Unsubscribe()

// Next: "tick" (immediately)
// Next: "tick" (after 1 second)
// Next: "tick" (after 2 seconds)
// Next: "tick" (after 3 seconds)
```

### With Take for limited repetitions

```go
source := ro.Just(1, 2, 3)
obs := ro.Pipe[int, int](
    RepeatWithInterval(source, 500*time.Millisecond),
    ro.Take[int](10),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(3000 * time.Millisecond)
sub.Unsubscribe()

// Next: 1, 2, 3 (immediately)
// Next: 1, 2, 3 (after 500ms)
// Next: 1, 2, 3 (after 1000ms)
// Next: 1 (after 1500ms)
// Completed (after taking 10 items)
```

### With complex sequences

```go
source := ro.Pipe[string, string](
    ro.Just("A", "B", "C"),
)
obs := ro.RepeatWithInterval(source, 800*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[string]())
time.Sleep(3000 * time.Millisecond)
sub.Unsubscribe()

// Next: A, B, C (immediately)
// Next: A, B, C (after 800ms)
// Next: A, B, C (after 1600ms)
// Next: A, B, C (after 2400ms)
```

### For periodic polling

```go
getTime := ro.Start(func() time.Time { return time.Now() })
obs := ro.RepeatWithInterval(getTime, 1*time.Second)

sub := obs.Subscribe(ro.PrintObserver[time.Time]())
time.Sleep(3500 * time.Millisecond)
sub.Unsubscribe()

// Next: <current time> (immediately)
// Next: <current time + 1s> (after 1 second)
// Next: <current time + 2s> (after 2 seconds)
// Next: <current time + 3s> (after 3 seconds)
```

### With error handling

```go
source := ro.Just(1, 2)
obs := ro.RepeatWithInterval(source, 500*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(2000 * time.Millisecond)
sub.Unsubscribe()

// Will repeat successfully:
// Next: 1, 2 (immediately)
// Next: 1, 2 (after 500ms)
// Next: 1, 2 (after 1000ms)
// Next: 1, 2 (after 1500ms)
```