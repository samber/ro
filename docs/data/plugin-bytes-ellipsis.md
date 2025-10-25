---
name: Ellipsis
slug: ellipsis
sourceRef: plugins/bytes/operator_ellipsis.go#L38
type: plugin
category: bytes
signatures:
  - "func Ellipsis[T ~[]byte](length int)"
playUrl: https://go.dev/play/p/zrihQpx4RFE
variantHelpers:
  - plugin#bytes#ellipsis
similarHelpers:
  - plugin#strings#ellipsis
position: 30
---

Truncates the string to specified length and appends "..." if longer.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("This is a very long string")),
    robytes.Ellipsis[[]byte](10),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [84 104 105 115 32 105 115 32 46 46 46]
// Completed
```