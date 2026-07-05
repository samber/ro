---
name: Trunc
slug: trunc
sourceRef: operator_math.go#L796
type: core
category: math
signatures:
  - "func Trunc()"
playUrl: https://go.dev/play/p/SpVO4Xmwfo0
variantHelpers:
  - core#math#trunc
similarHelpers:
  - core#math#round
  - core#math#floor
  - core#math#ceil
position: 31
---

Returns the integer value of each float64 emitted by the source Observable, truncating toward zero.

```go
obs := ro.Pipe[float64, float64](
    ro.Just(1.7, -1.7, 2.3, -2.3),
    ro.Trunc(),
)

sub := obs.Subscribe(ro.PrintObserver[float64]())
defer sub.Unsubscribe()

// Next: 1
// Next: -1
// Next: 2
// Next: -2
// Completed
```
