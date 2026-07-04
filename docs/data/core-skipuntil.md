---
name: SkipUntil
slug: skipuntil
sourceRef: operator_filter.go#L302
type: core
category: filtering
signatures:
  - "func SkipUntil[T any, S any](signal Observable[S])"
playUrl: ""
variantHelpers:
  - core#filtering#skipuntil
similarHelpers:
  - core#filtering#skip
  - core#filtering#skipwhile
  - core#filtering#takeuntil
position: 92
---

Skips items emitted by the source Observable until a signal Observable emits.

```go
signal := ro.Timer(200 * time.Millisecond)

obs := ro.Pipe[int, int](
    ro.Interval(50 * time.Millisecond),
    ro.SkipUntil[int](signal),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(400 * time.Millisecond)
sub.Unsubscribe()

// (items emitted before 200ms are skipped)
// Next: 4
// Next: 5
// Next: 6
// ...
```
