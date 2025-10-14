---
name: SnakeCase
slug: snakecase
sourceRef: plugins/strings/operator_snakecase.go#L33
type: plugin
category: strings
signatures:
  - "func SnakeCase[T ~string]()"
playUrl: ""
variantHelpers:
  - plugin#strings#snakecase
similarHelpers:
  - plugin#bytes#snakecase
position: 40
---

Converts string to snake case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe(
    ro.Just("HelloWorldWorld"),
    rostrings.SnakeCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello_world_world
// Completed
```