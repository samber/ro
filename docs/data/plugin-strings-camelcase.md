---
name: CamelCase
slug: camelcase
sourceRef: plugins/strings/operator_camelcase.go#L37
type: plugin
category: strings
signatures:
  - "func CamelCase[T ~string]()"
playUrl: ""
variantHelpers:
  - plugin#strings#camelcase
similarHelpers:
  - plugin#bytes#camelcase
position: 0
---

Converts string to camel case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("hello_world_world"),
    rostrings.CamelCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: helloWorldWorld
// Completed
```