---
name: KebabCase
slug: kebabcase
sourceRef: plugins/bytes/operator_kebabcase.go#L58
type: plugin
category: bytes
signatures:
  - "func KebabCase[T ~[]byte]()"
playUrl:
variantHelpers:
  - plugin#bytes#kebabcase
similarHelpers: []
position: 10
---

Converts the string to kebab case.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe(
    ro.Just([]byte("HelloWorldWorld")),
    robytes.KebabCase[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 45 119 111 114 108 100 45 119 111 114 108 100]
// Completed
```
