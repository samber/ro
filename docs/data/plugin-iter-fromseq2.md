---
name: FromSeq2
slug: fromseq2
sourceRef: plugins/iter/source.go#L36
type: plugin
category: iter
signatures:
  - "func FromSeq2[K, V any](iterator iter.Seq2[K, V])"
playUrl: https://go.dev/play/p/hHcR9l5TE0Q
variantHelpers:
  - plugin#iter#fromseq2
similarHelpers:
  - plugin#iter#toseq2
  - plugin#iter#fromseq
position: 10
---

Creates an observable from a Go iter.Seq2 iterator.

```go
import (
    "iter"

    "github.com/samber/ro"
    roiter "github.com/samber/ro/plugins/iter"
    "github.com/samber/lo"
)

m := map[string]int{"a": 1, "b": 2, "c": 3}
iterator := func(yield func(string, int) bool) {
    for k, v := range m {
        if !yield(k, v) {
            break
        }
    }
}

obs := roiter.FromSeq2(iterator)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[string, int]]())
defer sub.Unsubscribe()

// Next: {Data1: a Data2: 1}
// Next: {Data1: b Data2: 2}
// Next: {Data1: c Data2: 3}
// Completed
```