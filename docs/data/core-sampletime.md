---
name: SampleTime
slug: sampletime
sourceRef: operator_transformations.go#L771
type: core
category: transformation
signatures:
  - "func SampleTime[T any](interval time.Duration)"
playUrl: https://go.dev/play/p/PcPo4lE9-_T
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
obs := ro.Pipe[int64, int64](
    ro.Interval(10*time.Millisecond),
    ro.SampleTime[int64](25*time.Millisecond),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

time.Sleep(100 * time.Millisecond)

// Next: 1
// Next: 3
// Next: 6
// Next: 8
```
