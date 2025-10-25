---
name: Encode
slug: encode
sourceRef: plugins/encoding/base64/operator.go#L32
type: plugin
category: encoding-base64
signatures:
  - "func Encode[T ~[]byte](encoder *base64.Encoding)"
playUrl: https://go.dev/play/p/cFIbfAruPwz
variantHelpers:
  - plugin#encoding-base64#encode
similarHelpers:
  - plugin#encoding-base64#decode
position: 0
---

Encodes input into a base64 string.

```go
import (
    "encoding/base64"

    "github.com/samber/ro"
    robase64 "github.com/samber/ro/plugins/encoding/base64"
)

obs := ro.Pipe[[]byte, string](
    ro.Just([]byte("hello world")),
    robase64.Encode[[]byte](base64.StdEncoding),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: aGVsbG8gd29ybGQ=
// Completed
```