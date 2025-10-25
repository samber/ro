---
name: RangeWithInterval
slug: rangewithinterval
sourceRef: operator_creation.go#L168
type: core
category: creation
signatures:
  - "func RangeWithInterval(start int64, end int64, interval time.Duration)"
playUrl: https://go.dev/play/p/ykwKPLNquL9
variantHelpers:
  - core#creation#rangewithinterval
similarHelpers:
  - core#creation#range
  - core#creation#rangewithstep
  - core#creation#rangewithstepandinterval
  - core#creation#interval
  - core#creation#timer
position: 10
---

Creates an Observable that emits a sequence of numbers within a specified range with a time interval between emissions.

```go
obs := ro.RangeWithInterval(1, 5, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 1 (after 0ms)
// Next: 2 (after 100ms)
// Next: 3 (after 200ms)
// Next: 4 (after 300ms)
// Next: 5 (after 400ms)
// Completed
```

### Negative range with interval

```go
obs := ro.RangeWithInterval(5, 1, 100*time.Millisecond)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 5 (after 0ms)
// Next: 4 (after 100ms)
// Next: 3 (after 200ms)
// Next: 2 (after 300ms)
// Next: 1 (after 400ms)
// Completed
```