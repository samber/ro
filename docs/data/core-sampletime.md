---
name: SampleTime
slug: sampletime
sourceRef: operator_transformations.go#L771
type: core
category: transformation
signatures:
  - "func SampleTime[T any](interval time.Duration)"
playUrl: ""
variantHelpers:
  - core#transformation#sampletime
similarHelpers:
  - core#transformation#samplewhen
  - core#transformation#throttletime
  - core#transformation#throttlewhen
position: 91
---

Emits the most recently emitted item from the source Observable at regular time intervals.

```go
obs := ro.Pipe[int, int](
    ro.Interval(30 * time.Millisecond),
    ro.SampleTime[int](100 * time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
time.Sleep(350 * time.Millisecond)
sub.Unsubscribe()

// Emits the latest value every 100ms
```
