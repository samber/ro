---
name: FromSeq
slug: fromseq
sourceRef: plugins/iter/source.go#L26
type: plugin
category: iter
signatures:
  - "func FromSeq[T any](iterator iter.Seq[T])"
playUrl: https://go.dev/play/p/VwIX8SoYa9V
variantHelpers:
  - plugin#iter#fromseq
similarHelpers:
  - plugin#iter#toseq
  - plugin#iter#fromseq2
position: 0
---

Creates an observable from a Go iter.Seq iterator.

```go
import (
    "iter"

    "github.com/samber/ro"
    roiter "github.com/samber/ro/plugins/iter"
)

slice := []int{1, 2, 3, 4, 5}
iterator := func(yield func(int) bool) {
    for _, v := range slice {
        if !yield(v) {
            break
        }
    }
}

obs := roiter.FromSeq(iterator)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Next: 3
// Next: 4
// Next: 5
// Completed
```