---
name: Capitalize
slug: capitalize
sourceRef: plugins/bytes/operator_capitalize.go#L29
type: plugin
category: bytes
signatures:
  - "func Capitalize[T ~[]byte]()"
playUrl: https://go.dev/play/p/qc7UDCtJM0n
variantHelpers:
  - plugin#bytes#capitalize
similarHelpers:
  - plugin#strings#capitalize
position: 20
---

Capitalizes the first letter of the string.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello world")),
    robytes.Capitalize[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [72 101 108 108 111 32 119 111 114 108 100]
// Completed
```