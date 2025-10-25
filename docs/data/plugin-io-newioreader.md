---
name: NewIOReader
slug: newioreader
sourceRef: plugins/io/source.go#L29
type: plugin
category: io
signatures:
  - "func NewIOReader(reader io.Reader)"
playUrl: https://go.dev/play/p/IvjWBKDHYHM
variantHelpers:
  - plugin#io#newioreader
similarHelpers:
  - plugin#io#newioreaderline
  - plugin#io#newstdreader
position: 0
---

Creates an observable that reads data from an io.Reader in chunks.

```go
import (
    "io"
    "strings"

    "github.com/samber/ro"
    roio "github.com/samber/ro/plugins/io"
)

data := strings.NewReader("Hello, World!")
obs := roio.NewIOReader(data)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [72 101 108 108 111 44 32 87 111 114 108 100 33]
// Completed
```