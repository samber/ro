---
name: Ellipsis
slug: ellipsis
sourceRef: plugins/strings/operator_ellipsis.go#L38
type: plugin
category: strings
signatures:
  - "func Ellipsis[T ~string](length int)"
playUrl: ""
variantHelpers:
  - plugin#strings#ellipsis
similarHelpers:
  - plugin#bytes#ellipsis
position: 30
---

Truncates string to length with ellipsis.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe(
    ro.Just("This is a very long string"),
    rostrings.Ellipsis[string](10),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: This is...
// Completed
```