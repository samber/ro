---
name: SkipUntil
slug: skipuntil
sourceRef: operator_filter.go#L302
type: core
category: filtering
signatures:
  - "func SkipUntil[T any, S any](signal Observable[S])"
playUrl: https://go.dev/play/p/tAwg2LT3Hqn
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

obs := ro.Pipe[int64, int64](
    ro.Interval(50 * time.Millisecond),
    ro.SkipUntil[int64](signal),
    ro.Take[int64](3),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// (items emitted before 200ms are skipped)
// Next: 3
// Next: 4
// Next: 5
// Completed
```
