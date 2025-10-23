---
name: CamelCase
slug: camelcase
sourceRef: plugins/bytes/operator_camelcase.go#L37
type: plugin
category: bytes
signatures:
  - "func CamelCase[T ~[]byte]()"
playUrl: ""
variantHelpers:
  - plugin#bytes#camelcase
similarHelpers:
  - plugin#strings#camelcase
position: 0
---

Converts the string to camel case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello_world_world")),
    robytes.CamelCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 87 111 114 108 100 87 111 114 108 100]
// Completed
```