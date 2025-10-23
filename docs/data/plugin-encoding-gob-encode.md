---
name: Encode
slug: encode
sourceRef: plugins/encoding/gob/operator.go#L25
type: plugin
category: encoding-gob
signatures:
  - "func Encode[T any]()"
playUrl: ""
variantHelpers:
  - plugin#encoding-gob#encode
similarHelpers:
  - plugin#encoding-json#marshal
  - plugin#encoding-json#marshalv2
position: 0
---

Encodes values to gob binary format.

```go
import (
    "github.com/samber/ro"
    rogob "github.com/samber/ro/plugins/encoding/gob"
)

obs := ro.Pipe[int, []byte](
    ro.Just(42),
    rogob.Encode[int](),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [3 4 0 84]
// Completed
```