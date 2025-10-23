---
name: Log
slug: log
sourceRef: plugins/observability/logrus/operator.go#L25
type: plugin
category: logger-logrus
signatures:
  - "func Log[T any](logger *logrus.Logger, level logrus.Level)"
playUrl: ""
variantHelpers:
  - plugin#logger-logrus#log
  - plugin#logger-logrus#logwithnotification
similarHelpers:
  - plugin#logger-logrus#logwithnotification
  - plugin#logger-logrus#fatalonerror
position: 0
---

Logs with logrus.

```go
import (
    "github.com/samber/ro"
    rologrus "github.com/samber/ro/plugins/observability/logrus"
    "github.com/sirupsen/logrus"
)

logger := logrus.New()
logger.SetLevel(logrus.InfoLevel)

obs := ro.Pipe[string, string](
    ro.Just("message 1", "message 2"),
    rologrus.Log[string](logger, logrus.InfoLevel),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Logs: message 1
// Logs: message 2
// Next: message 1
// Next: message 2
// Completed
```