---
name: SortFunc
slug: sortfunc
sourceRef: plugins/sort/operator.go#L61
type: plugin
category: sort
signatures:
  - "func SortFunc[T comparable](cmp func(a, b T) int)"
playUrl: https://go.dev/play/p/SUtwR8m5gD6
variantHelpers:
  - plugin#sort#sortfunc
similarHelpers:
  - plugin#sort#sort
  - plugin#sort#sortstablefunc
position: 10
---

Sorts values using comparison function.

```go
import (
    "strings"

    "github.com/samber/ro"
    rosort "github.com/samber/ro/plugins/sort"
)

obs := ro.Pipe[string, string](
    ro.Just("banana", "apple", "cherry"),
    rosort.SortFunc[string](func(a, b string) int {
        return strings.Compare(a, b)
    }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: apple
// Next: banana
// Next: cherry
// Completed
```