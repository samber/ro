---
name: SampleWhen
slug: samplewhen
sourceRef: operator_transformations.go#L707
type: core
category: transformation
signatures:
  - "func SampleWhen[T any, t any](tick Observable[t])"
playUrl: https://go.dev/play/p/tr4FEd-CSce
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
source := ro.Interval(10 * time.Millisecond)
sampler := ro.Interval(30 * time.Millisecond)

obs := ro.Pipe[int64, int64](
    source,
    ro.SampleWhen[int64](sampler),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

time.Sleep(100 * time.Millisecond)

// Next: 1
// Next: 4
// Next: 7
```
