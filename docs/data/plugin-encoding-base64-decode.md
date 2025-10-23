---
name: Decode
slug: decode
sourceRef: plugins/encoding/base64/operator.go#L46
type: plugin
category: encoding-base64
signatures:
  - "func Decode[T ~string](encoder *base64.Encoding)"
playUrl: ""
variantHelpers:
  - plugin#encoding-base64#decode
similarHelpers:
  - plugin#encoding-base64#encode
position: 10
---

Decodes input from a base64 string.

```go
import (
    "encoding/base64"

    "github.com/samber/ro"
    robase64 "github.com/samber/ro/plugins/encoding/base64"
)

obs := ro.Pipe[string, []byte](
    ro.Just("aGVsbG8gd29ybGQ="),
    robase64.Decode[string](base64.StdEncoding),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 32 119 111 114 108 100]
// Completed
```