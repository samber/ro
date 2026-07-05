---
name: ThrottleTime
slug: throttletime
sourceRef: operator_transformations.go#L829
type: core
category: transformation
signatures:
  - "func ThrottleTime[T any](interval time.Duration)"
playUrl: https://go.dev/play/p/ExdxZiAE0Eu
variantHelpers:
  - core#transformation#throttletime
similarHelpers:
  - core#transformation#throttlewhen
  - core#transformation#sampletime
  - core#transformation#samplewhen
position: 93
---

Emits a value from the source Observable, then ignores subsequent source values for a fixed time duration.

```go
obs := ro.Pipe[int64, int64](
    ro.Interval(10*time.Millisecond),
    ro.ThrottleTime[int64](30*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

time.Sleep(100 * time.Millisecond)

// Next: 3
// Next: 7
```
