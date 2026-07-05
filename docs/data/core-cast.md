---
name: Cast
slug: cast
sourceRef: operator_transformations.go#L236
type: core
category: transformation
signatures:
  - "func Cast[T any, U any]()"
playUrl: https://go.dev/play/p/XUdqodfFyT6
variantHelpers:
  - core#transformation#cast
similarHelpers:
  - core#transformation#map
position: 110
---

Casts each item emitted by an Observable to a target type. Panics if the item cannot be cast to the target type.

```go
obs := ro.Pipe[any, int](
    ro.Just[any](1, 2, 3),
    ro.Cast[any, int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Completed
```
