---
name: EndWith
slug: endwith
sourceRef: operator_combining.go#L949
type: core
category: combining
signatures:
  - "func EndWith[T any](suffixes ...T)"
playUrl: https://go.dev/play/p/MfSijaXU7sq
variantHelpers:
  - core#combining#endwith
similarHelpers:
  - core#combining#startwith
position: 76
---

Emits additional values after the source Observable completes.

```go
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    ro.EndWith(4, 5),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Completed
```
