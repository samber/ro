---
name: ThrottleWhen
slug: throttlewhen
sourceRef: operator_transformations.go#L780
type: core
category: transformation
signatures:
  - "func ThrottleWhen[T any, t any](tick Observable[t])"
playUrl: ""
variantHelpers:
  - core#transformation#throttlewhen
similarHelpers:
  - core#transformation#throttletime
  - core#transformation#samplewhen
  - core#transformation#sampletime
position: 92
---

Emits a value from the source Observable, then ignores subsequent source values for a duration determined by the tick Observable.

```go
tick := ro.Interval(100 * time.Millisecond)

obs := ro.Pipe[int, int](
    ro.Interval(20 * time.Millisecond),
    ro.ThrottleWhen[int](tick),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(350 * time.Millisecond)
sub.Unsubscribe()

// Emits first item, then throttles until next tick
```
