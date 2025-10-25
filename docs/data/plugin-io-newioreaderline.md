---
name: NewIOReaderLine
slug: newioreaderline
sourceRef: plugins/io/source.go#L54
type: plugin
category: io
signatures:
  - "func NewIOReaderLine(reader io.Reader)"
playUrl: https://go.dev/play/p/m9xsZX9z-dP
variantHelpers:
  - plugin#io#newioreaderline
similarHelpers:
  - plugin#io#newioreader
  - plugin#io#newstdreaderline
position: 10
---

Creates an observable that reads lines from an io.Reader.

```go
import (
    "strings"

    "github.com/samber/ro"
    roio "github.com/samber/ro/plugins/io"
)

data := strings.NewReader("line1\nline2\nline3")
obs := roio.NewIOReaderLine(data)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [108 105 110 101 49]
// Next: [108 105 110 101 50]
// Next: [108 105 110 101 51]
// Completed
```