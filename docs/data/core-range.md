---
name: Range
slug: range
sourceRef: operator_creation.go#L168
type: core
category: creation
signatures:
  - "func Range(start int64, end int64)"
playUrl: https://go.dev/play/p/5XAXfNrtJm2
variantHelpers:
  - core#creation#range
similarHelpers:
  - core#creation#rangewithstep
  - core#creation#rangewithinterval
  - core#creation#rangewithstepandinterval
  - core#creation#interval
  - core#creation#timer
position: 10
---

Creates an Observable that emits a sequence of numbers within a specified range.

```go
obs := ro.Range(1, 5)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Completed
```

### Negative range

```go
obs := ro.Range(5, 1)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 5
// Next: 4
// Next: 3
// Next: 2
// Next: 1
// Completed
```

### Single value range

```go
obs := ro.Range(5, 5)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 5
// Completed
```