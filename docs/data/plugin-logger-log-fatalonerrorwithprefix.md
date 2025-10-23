---
name: FatalOnErrorWithPrefix
slug: fatalonerrorwithprefix
sourceRef: plugins/observability/log/operator.go#L59
type: plugin
category: logger-log
signatures:
  - "func FatalOnErrorWithPrefix[T any](prefix string)"
playUrl: ""
variantHelpers:
  - plugin#logger-log#fatalonerrorwithprefix
similarHelpers:
  - plugin#logger-log#logwithprefix
  - plugin#logger-log#fatalonerror
position: 3
---

Terminates the application on error with prefixed logging.

```go
import (
    "errors"

    "github.com/samber/ro"
    rolog "github.com/samber/ro/plugins/observability/log"
)

obs := ro.Pipe[int, int](
    ro.Just(1, 2),
    ro.Throw[int](errors.New("fatal error")),
    rolog.FatalOnErrorWithPrefix[int]("Critical"),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Logs: Critical ro.Error: fatal error
// Application terminates with fatal error
```