---
name: LogWithPrefix
slug: logwithprefix
sourceRef: plugins/observability/log/operator.go#L31
type: plugin
category: logger-log
signatures:
  - "func LogWithPrefix[T any](prefix string)"
playUrl: ""
variantHelpers:
  - plugin#logger-log#logwithprefix
similarHelpers:
  - plugin#logger-log#log
  - plugin#logger-log#fatalonerrorwithprefix
position: 1
---

Logs observable events with a custom prefix.

```go
import (
    "github.com/samber/ro"
    rolog "github.com/samber/ro/plugins/observability/log"
)

obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3),
    rolog.LogWithPrefix[int]("MyStream"),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Logs: MyStream ro.Next: 1
// Logs: MyStream ro.Next: 2
// Logs: MyStream ro.Next: 3
// Next: 1
// Next: 2
// Next: 3
// Completed
```