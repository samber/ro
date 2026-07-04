---
name: ThrottleTime
slug: throttletime
sourceRef: operator_transformations.go#L829
type: core
category: transformation
signatures:
  - "func ThrottleTime[T any](interval time.Duration)"
playUrl: ""
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
obs := ro.Pipe[int, int](
    ro.Interval(20 * time.Millisecond),
    ro.ThrottleTime[int](100 * time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(350 * time.Millisecond)
sub.Unsubscribe()

// Emits first item, then ignores items for 100ms
```
