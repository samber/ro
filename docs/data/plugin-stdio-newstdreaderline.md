---
name: NewStdReaderLine
slug: newstdreaderline
sourceRef: plugins/stdio/source.go#L86
type: plugin
category: stdio
signatures:
  - "func NewStdReaderLine()"
playUrl: ""
variantHelpers:
  - plugin#io#newstdreaderline
similarHelpers:
  - plugin#io#newioreaderline
  - plugin#io#newstdreader
position: 30
---

Creates an observable that reads lines from standard input.

```go
import (
    "github.com/samber/ro"
    rostdio "github.com/samber/ro/plugins/stdio"
)

obs := rostdio.NewStdReaderLine()

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Reads lines from stdin when user enters data
// Next: [104 101 108 108 111]  // if user types "hello"
// Completed
```