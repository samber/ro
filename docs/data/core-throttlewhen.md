---
name: ThrottleWhen
slug: throttlewhen
sourceRef: operator_transformations.go#L780
type: core
category: transformation
signatures:
  - "func ThrottleWhen[T any, t any](tick Observable[t])"
playUrl: https://go.dev/play/p/q3ISV03EL3q
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
source := ro.Interval(10 * time.Millisecond)
tick := ro.Interval(30 * time.Millisecond)

obs := ro.Pipe[int64, int64](
    source,
    ro.ThrottleWhen[int64](tick),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

time.Sleep(100 * time.Millisecond)

// Next: 2
// Next: 6
```
