---
name: NewStdWriter
slug: newstdwriter
sourceRef: plugins/io/sink.go#L59
type: plugin
category: io
signatures:
  - "func NewStdWriter()"
playUrl: ""
variantHelpers:
  - plugin#io#newstdwriter
similarHelpers:
  - plugin#io#newiowriter
position: 50
---

Creates an operator that writes byte arrays to standard output and returns the count of written bytes.

```go
import (
    "github.com/samber/ro"
    roio "github.com/samber/ro/plugins/io"
)

obs := ro.Pipe[[]byte, int](
    ro.Just([]byte("Hello, World!")),
    roio.NewStdWriter(),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Hello, World! (written to stdout)
// Next: 13
// Completed
```