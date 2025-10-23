---
name: Average
slug: average
sourceRef: operator_math.go#L24
type: core
category: math
signatures:
  - "func Average[T Numeric]()"
playUrl: ""
variantHelpers:
  - core#math#average
similarHelpers: []
position: 0
---

Calculates the average of the values emitted by the source Observable. It emits the average when the source completes. If the source is empty, it emits NaN.

```go
obs := ro.Pipe[int, float64](
    ro.Just(1, 2, 3, 4, 5),
    ro.Average[int](),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 3
// Completed
```