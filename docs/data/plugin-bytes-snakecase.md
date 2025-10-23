---
name: SnakeCase
slug: snakecase
sourceRef: plugins/bytes/operator_snakecase.go#L33
type: plugin
category: bytes
signatures:
  - "func SnakeCase[T ~[]byte]()"
playUrl: ""
variantHelpers:
  - plugin#bytes#snakecase
similarHelpers:
  - plugin#strings#snakecase
position: 60
---

Converts the string to snake case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("HelloWorldWorld")),
    robytes.SnakeCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 95 119 111 114 108 100 95 119 111 114 108 100]
// Completed
```