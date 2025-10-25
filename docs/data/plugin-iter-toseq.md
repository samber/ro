---
name: ToSeq
slug: toseq
sourceRef: plugins/iter/sink.go#L24
type: plugin
category: iter
signatures:
  - "func ToSeq[T any](source ro.Observable[T]) iter.Seq[T]"
playUrl: https://go.dev/play/p/g8R8jb35LAs
variantHelpers:
  - plugin#iter#toseq
similarHelpers:
  - plugin#iter#fromseq
  - plugin#iter#toseq2
position: 20
---

Converts an observable to a Go iter.Seq iterator.

```go
import (
    "fmt"
    "iter"

    "github.com/samber/ro"
    roiter "github.com/samber/ro/plugins/iter"
)

obs := ro.Just(1, 2, 3, 4, 5)
iterator := roiter.ToSeq(obs)

for v := range iterator {
    fmt.Println("Value:", v)
}

// 
// Value: 1
// Value: 2
// Value: 3
// Value: 4
// Value: 5
```