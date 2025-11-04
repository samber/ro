---
name: NewStdReader
slug: newstdreader
sourceRef: plugins/stdio/source.go#L82
type: plugin
category: stdio
signatures:
  - "func NewStdReader()"
playUrl: https://go.dev/play/p/YDjiTqvKbcl
variantHelpers:
  - plugin#io#newstdreader
similarHelpers:
  - plugin#io#newioreader
  - plugin#io#newstdreaderline
position: 20
---

Creates an observable that reads data from standard input.

```go
import (
    "github.com/samber/ro"
    rostdio "github.com/samber/ro/plugins/stdio"
)

obs := rostdio.NewStdReader()

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Reads from stdin when data is available
// Next: [104 101 108 108 111]  // if user types "hello"
// Completed
```