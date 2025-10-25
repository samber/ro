---
name: Sort
slug: sort
sourceRef: plugins/sort/operator.go#L40
type: plugin
category: sort
signatures:
  - "func Sort[T cmp.Ordered](cmp func(a, b T) int)"
playUrl: https://go.dev/play/p/Jem9ufkfmNR
variantHelpers:
  - plugin#sort#sort
similarHelpers:
  - plugin#sort#sortfunc
  - plugin#sort#sortstablefunc
position: 0
---

Sorts ordered values (loads all into memory).

```go
import (
    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

obs := ro.Pipe[int, int](
    ro.Just(3, 1, 4, 1, 5, 9, 2, 6),
    rosort.Sort[int](func(a, b int) int { return a - b }),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Next: 6
// Next: 9
// Completed
```