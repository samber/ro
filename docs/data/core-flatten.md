---
name: Flatten
slug: flatten
sourceRef: operator_transformations.go#L213
type: core
category: transformation
signatures:
  - "func Flatten[T any]()"
playUrl: ""
variantHelpers:
  - core#transformation#flatten
similarHelpers:
  - core#transformation#flatmap
  - core#transformation#mergeall
position: 11
---

Flattens an Observable of slices into an Observable of individual items.

```go
obs := ro.Pipe[[]int, int](
    ro.Just([]int{1, 2, 3}, []int{4, 5, 6}),
    ro.Flatten[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Next: 6
// Completed
```
