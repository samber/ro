---
name: KebabCase
slug: kebabcase
sourceRef: plugins/strings/operator_kebabcase.go#L33
type: plugin
category: strings
signatures:
  - "func KebabCase[T ~string]()"
playUrl: https://go.dev/play/p/yAbSRKFl4pS
variantHelpers:
  - plugin#strings#kebabcase
similarHelpers:
  - plugin#strings#snakecase
  - plugin#strings#camelcase
position: 20
---

Converts string to kebab case.

```go
import (
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, string](
    ro.Just("HelloWorldTest"),
    rostrings.KebabCase[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello-world-test
// Completed
```