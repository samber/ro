---
name: NewStdWriter
slug: newstdwriter
sourceRef: plugins/io/sink.go#L59
type: plugin
category: stdio
signatures:
  - "func NewStdWriter()"
playUrl: https://go.dev/play/p/tj1fyTjkCDn
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
    rostdio "github.com/samber/ro/plugins/stdio"
)

obs := ro.Pipe[[]byte, int](
    ro.Just([]byte("Hello, World!")),
    rostdio.NewStdWriter(),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Hello, World! (written to stdout)
// Next: 13
// Completed
```