---
name: FormatInt
slug: formatint
sourceRef: plugins/strconv/operator.go#L176
type: plugin
category: strconv
signatures:
  - "func FormatInt[T ~string](base int)"
playUrl: ""
variantHelpers:
  - plugin#strconv#formatint
similarHelpers:
  - plugin#strconv#formatuint
  - plugin#strconv#formatfloat
  - plugin#strconv#parseint
  - plugin#strconv#itoa
position: 80
---

Converts each int64 emitted by the source Observable to its string representation in the given base.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[int64, string](
    ro.Just[int64](255, 16, 8),
    rostrconv.FormatInt[string](16),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: ff
// Next: 10
// Next: 8
// Completed
```
