---
name: SkipLast
slug: skiplast
sourceRef: operator_filter.go#L262
type: core
category: filtering
signatures:
  - "func SkipLast[T any](count int)"
playUrl: ""
variantHelpers:
  - core#filtering#skiplast
similarHelpers:
  - core#filtering#skip
  - core#filtering#takelast
position: 262
---

Suppresses the last n items emitted by an Observable.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    ro.SkipLast[int](2),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```
