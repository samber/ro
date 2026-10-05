---
name: TakeUntil
slug: takeuntil
sourceRef: operator_filter.go#L544
type: core
category: filtering
signatures:
  - "func TakeUntil[T any, S any](signal Observable[S])"
playUrl: https://go.dev/play/p/moJPw7uKjrz
variantHelpers:
  - core#filtering#takeuntil
similarHelpers:
  - core#filtering#take
  - core#filtering#takewhile
  - core#filtering#skipuntil
position: 21
---

Emits items from the source Observable until a signal Observable emits.

The signal is subscribed before the source. If the signal errors, the error is forwarded downstream. If the signal completes without emitting, all items are emitted.

```go
signal := ro.Timer(200 * time.Millisecond)

obs := ro.Pipe[int64, int64](
    ro.Interval(50 * time.Millisecond),
    ro.TakeUntil[int64](signal),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 0
// Next: 1
// Next: 2
// Next: 3
// Completed (after ~200ms)
```
