---
name: PascalCase
slug: pascalcase
sourceRef: plugins/bytes/operator_pascalcase.go#L33
type: plugin
category: bytes
signatures:
  - "func PascalCase[T ~[]byte]()"
playUrl: ""
variantHelpers:
  - plugin#bytes#pascalcase
similarHelpers:
  - plugin#strings#pascalcase
position: 40
---

Converts the string to pascal case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello_world_world")),
    robytes.PascalCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [72 101 108 108 111 87 111 114 108 100 87 111 114 108 100]
// Completed
```