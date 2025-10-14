---
name: ToSeq2
slug: toseq2
sourceRef: plugins/iter/sink.go#L71
type: plugin
category: iter
signatures:
  - "func ToSeq2[T any](source ro.Observable[T]) iter.Seq2[int, T]"
playUrl: ""
variantHelpers:
  - plugin#iter#toseq2
similarHelpers:
  - plugin#iter#fromseq2
  - plugin#iter#toseq
position: 30
---

Converts an observable to a Go iter.Seq2 iterator with index.

```go
import (
    "fmt"
    "iter"

    "github.com/samber/ro"
    roiter "github.com/samber/ro/plugins/iter"
)

obs := ro.Just("a", "b", "c")
iterator := roiter.ToSeq2(obs)

for i, v := range iterator {
    fmt.Printf("Index: %d, Value: %s\n", i, v)
}

// 
// Index: 0, Value: a
// Index: 1, Value: b
// Index: 2, Value: c
```