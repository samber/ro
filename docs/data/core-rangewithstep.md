---
name: RangeWithStep
slug: rangewithstep
sourceRef: operator_creation.go#L168
type: core
category: creation
signatures:
  - "func RangeWithStep(start float64, end float64, step float64)"
playUrl: https://go.dev/play/p/61lr9W1Mkf0
variantHelpers:
  - core#creation#rangewithstep
similarHelpers:
  - core#creation#range
  - core#creation#rangewithinterval
  - core#creation#rangewithstepandinterval
  - core#creation#interval
  - core#creation#timer
position: 10
---

Creates an Observable that emits a sequence of numbers within a specified range with a custom step.

```go
obs := ro.RangeWithStep(1, 10, 2)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 1
// Next: 3
// Next: 5
// Next: 7
// Next: 9
// Completed
```

### Fractional range

```go
obs := ro.RangeWithStep(0.5, 2.5, 0.5)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 0.5
// Next: 1
// Next: 1.5
// Next: 2
// Next: 2.5
// Completed
```

### Negative step

```go
obs := ro.RangeWithStep(10, 1, -2)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 10
// Next: 8
// Next: 6
// Next: 4
// Next: 2
// Completed
```