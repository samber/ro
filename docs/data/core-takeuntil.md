---
name: TakeUntil
slug: takeuntil
sourceRef: operator_filter.go#L516
type: core
category: filtering
signatures:
  - "func TakeUntil[T any, S any](signal Observable[S])"
playUrl: ""
variantHelpers:
  - core#filtering#takeuntil
similarHelpers:
  - core#filtering#take
  - core#filtering#takewhile
  - core#filtering#skipuntil
position: 21
---

Emits items from the source Observable until a signal Observable emits or completes.

```go
signal := ro.Timer(200 * time.Millisecond)

obs := ro.Pipe[int, int](
    ro.Interval(50 * time.Millisecond),
    ro.TakeUntil[int](signal),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 0
// Next: 1
// Next: 2
// Next: 3
// Completed (after ~200ms)
```
