---
name: FormatUint
slug: formatuint
sourceRef: plugins/strconv/operator.go#L190
type: plugin
category: strconv
signatures:
  - "func FormatUint[T ~string](base int)"
playUrl: ""
variantHelpers:
  - plugin#strconv#formatuint
similarHelpers:
  - plugin#strconv#formatint
  - plugin#strconv#formatfloat
  - plugin#strconv#parseuint
position: 90
---

Converts each uint64 emitted by the source Observable to its string representation in the given base.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[uint64, string](
    ro.Just[uint64](255, 1024, 0),
    rostrconv.FormatUint[string](2),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 11111111
// Next: 10000000000
// Next: 0
// Completed
```
