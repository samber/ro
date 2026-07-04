---
name: MapTo
slug: mapto
sourceRef: operator_transformations.go#L80
type: core
category: transformation
signatures:
  - "func MapTo[T any, R any](output R)"
playUrl: ""
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
    ro.Just(1, 2, 3, 4, 5),
    ro.MapTo[int, string]("hello"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello
// Next: hello
// Next: hello
// Next: hello
// Next: hello
// Completed
```
