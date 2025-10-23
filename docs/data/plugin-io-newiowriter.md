---
name: NewIOWriter
slug: newiowriter
sourceRef: plugins/io/sink.go#L26
type: plugin
category: io
signatures:
  - "func NewIOWriter(writer io.Writer)"
playUrl: ""
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
    roio "github.com/samber/ro/plugins/io"
)

var buf bytes.Buffer
obs := ro.Pipe[[]byte, int](
    ro.Just([]byte("Hello, "), []byte("World!")),
    roio.NewIOWriter(&buf),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 13
// Completed
```