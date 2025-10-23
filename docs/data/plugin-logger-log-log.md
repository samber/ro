---
name: Log
slug: log
sourceRef: plugins/observability/log/operator.go#L25
type: plugin
category: logger-log
signatures:
  - "func Log[T any]()"
playUrl: ""
variantHelpers:
  - plugin#logger-log#log
  - plugin#logger-log#logwithprefix
similarHelpers:
  - plugin#log#logwithprefix
position: 0
---

Logs observable events.

```go
import (
    "github.com/samber/ro"
    rolog "github.com/samber/ro/plugins/observability/log"
)

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    rolog.Log[int](),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Logs: [1]
// Logs: [2]
// Logs: [3]
// Next: 1
// Next: 2
// Next: 3
// Completed
```