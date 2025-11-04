---
name: NewIOWriter
slug: newiowriter
sourceRef: plugins/stdio/sink.go#L26
type: plugin
category: stdio
signatures:
  - "func NewIOWriter(writer io.Writer)"
playUrl: https://go.dev/play/p/LazCm43IDml
variantHelpers:
  - plugin#io#newiowriter
similarHelpers:
  - plugin#io#newstdwriter
position: 40
---

Creates an operator that writes byte arrays to an io.Writer and returns the count of written bytes.

```go
import (
    "bytes"

    "github.com/samber/ro"
    rostdio "github.com/samber/ro/plugins/stdio"
)

var buf bytes.Buffer
obs := ro.Pipe[[]byte, int](
    ro.Just([]byte("Hello, "), []byte("World!")),
    rostdio.NewIOWriter(&buf),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 13
// Completed
```