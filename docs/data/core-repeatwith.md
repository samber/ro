---
name: RepeatWith
slug: repeatwith
sourceRef: operator_utility.go#L396
type: core
category: utility
signatures:
  - "func RepeatWith[T any](count int64)"
playUrl: https://go.dev/play/p/fEKtAX9_nYe
variantHelpers:
  - core#utility#repeatwith
similarHelpers:
  - core#creation#repeat
  - core#creation#repeatwithinterval
position: 400
---

Repeats the source Observable a fixed number of times by re-subscribing after each completion.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.RepeatWith[int](2),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 1
// Next: 2
// Next: 3
// Completed
```
