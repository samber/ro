---
name: Words
slug: words
sourceRef: plugins/bytes/operator_words.go#L41
type: plugin
category: bytes
signatures:
  - "func Words[T ~[]byte]()"
playUrl: ""
variantHelpers:
  - plugin#bytes#words
similarHelpers:
  - plugin#strings#words
position: 70
---

Splits the string into words.

```go
import (
    "github.com/samber/ro"
    robytes "github.com/samber/ro/plugins/bytes"
)

obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello world from go")),
    robytes.Words[[]byte](),
)

sub := obs.Subscribe(ro.PrintObserver[[][]byte]())
defer sub.Unsubscribe()

// Next: [[104 101 108 108 111] [119 111 114 108 100] [102 114 111 109] [103 111]]
// Completed
```