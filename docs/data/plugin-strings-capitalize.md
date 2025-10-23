---
name: Capitalize
slug: capitalize
sourceRef: plugins/strings/operator_capitalize.go#L29
type: plugin
category: strings
signatures:
  - "func Capitalize[T ~string]()"
playUrl: ""
variantHelpers:
  - plugin#strings#capitalize
similarHelpers:
  - plugin#bytes#capitalize
position: 10
---

Capitalizes first letter of string.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("hello world"),
    rostrings.Capitalize[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Hello world
// Completed
```