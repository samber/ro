---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/log/operator.go#L47
type: plugin
category: logger-log
signatures:
  - "func FatalOnError[T any]()"
playUrl: ""
variantHelpers:
  - plugin#logger-log#fatalonerror
similarHelpers:
  - plugin#log#logwithprefix
position: 10
---

Calls fatal on error.

```go
import (
    "errors"

    "github.com/samber/ro"
    rolog "github.com/samber/ro/plugins/observability/log"
)

obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("fatal error")),
    rolog.FatalOnError[string](),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Fatal: fatal error
// (program exits)
```