---
name: SampleWhen
slug: samplewhen
sourceRef: operator_transformations.go#L707
type: core
category: transformation
signatures:
  - "func SampleWhen[T any, t any](tick Observable[t])"
playUrl: ""
variantHelpers:
  - core#transformation#samplewhen
similarHelpers:
  - core#transformation#sampletime
  - core#transformation#throttlewhen
  - core#transformation#throttletime
position: 90
---

Emits the most recently emitted item from the source Observable whenever a tick Observable emits.

```go
tick := ro.Interval(100 * time.Millisecond)

obs := ro.Pipe[int, int](
    ro.Interval(30 * time.Millisecond),
    ro.SampleWhen[int](tick),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(350 * time.Millisecond)
sub.Unsubscribe()

// Emits the latest value every ~100ms
```
