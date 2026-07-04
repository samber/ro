---
name: MapTo
slug: mapto
sourceRef: operator_transformations.go#L80
type: core
category: transformation
signatures:
  - "func MapTo[T any, R any](output R)"
playUrl: https://go.dev/play/p/Ghc5ar7GJag
variantHelpers:
  - core#transformation#mapto
similarHelpers:
  - core#transformation#map
  - core#transformation#maperr
position: 1
---

Maps every item emitted by an Observable to the same constant value.

```go
obs := ro.Pipe[int, string](
    ro.Just(1, 2, 3),
    ro.MapTo[int, string]("converted"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: converted
// Next: converted
// Next: converted
// Completed
```
