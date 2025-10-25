---
name: Decode
slug: decode
sourceRef: plugins/encoding/gob/operator.go#L33
type: plugin
category: encoding-gob
signatures:
  - "func Decode[T any]()"
playUrl: https://go.dev/play/p/NHf4jgVRNJt
variantHelpers:
  - plugin#encoding-gob#decode
similarHelpers:
  - plugin#encoding-json#unmarshal
  - plugin#encoding-json#unmarshalv2
position: 10
---

Decodes gob binary format to typed values.

```go
import (
    "github.com/samber/ro"
    rogob "github.com/samber/ro/plugins/encoding/gob"
)

encoded := []byte{37, 255, 141, 3, 1, 1, 6, 80, 101, 114, 115, 111, 110, 1, 255, 142, 0, 1, 2, 1, 4, 78, 97, 109, 101, 1, 12, 0, 1, 3, 65, 103, 101, 1, 4, 0, 0, 0, 12, 255, 142, 1, 5, 65, 108, 105, 99, 101, 1, 60, 0}

obs := ro.Pipe[[]byte, Person](
    ro.Just(encoded),
    rogob.Decode[Person](),
)

sub := obs.Subscribe(ro.PrintObserver[Person]())
defer sub.Unsubscribe()

// Next: {Alice 30}
// Completed
```