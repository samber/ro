---
name: RangeWithStepAndInterval
slug: rangewithstepandinterval
sourceRef: operator_creation.go#L168
type: core
category: creation
signatures:
  - "func RangeWithStepAndInterval(start float64, end float64, step float64, interval time.Duration)"
playUrl: https://go.dev/play/p/2FcaPDM5lF5
variantHelpers:
  - core#creation#rangewithstepandinterval
similarHelpers:
  - core#creation#range
  - core#creation#rangewithstep
  - core#creation#rangewithinterval
  - core#creation#interval
  - core#creation#timer
position: 10
---

Creates an Observable that emits a sequence of numbers within a specified range with a custom step and time interval between emissions.

```go
obs := ro.RangeWithStepAndInterval(1, 10, 2, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 1 (after 0ms)
// Next: 3 (after 100ms)
// Next: 5 (after 200ms)
// Next: 7 (after 300ms)
// Next: 9 (after 400ms)
// Completed
```

### Fractional range with interval

```go
obs := ro.RangeWithStepAndInterval(0.5, 2.5, 0.5, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.5 (after 0ms)
// Next: 1 (after 100ms)
// Next: 1.5 (after 200ms)
// Next: 2 (after 300ms)
// Next: 2.5 (after 400ms)
// Completed
```

### Negative step with interval

```go
obs := ro.RangeWithStepAndInterval(10, 1, -2, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 10 (after 0ms)
// Next: 8 (after 100ms)
// Next: 6 (after 200ms)
// Next: 4 (after 300ms)
// Next: 2 (after 400ms)
// Completed
```