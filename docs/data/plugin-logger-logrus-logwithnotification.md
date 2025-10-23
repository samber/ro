---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/logrus/operator.go#L39
type: plugin
category: logger-logrus
signatures:
  - "func LogWithNotification[T any](logger *logrus.Logger, level logrus.Level)"
playUrl: ""
variantHelpers:
  - plugin#logger-logrus#log
  - plugin#logger-logrus#logwithnotification
similarHelpers:
  - plugin#logger-logrus#log
  - plugin#logger-logrus#fatalonerror
position: 10
---

Logs with logrus with structured fields and notifications.

```go
import (
    "github.com/samber/ro"
    "github.com/sirupsen/logrus"
    rologrus "github.com/samber/ro/plugins/observability/logrus"
)

logger := logrus.New()
logger.SetLevel(logrus.InfoLevel)

obs := ro.Pipe[string, string](
    ro.Just("user login", "data processing", "task completed"),
    rologrus.LogWithNotification[string](logger, logrus.InfoLevel),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Logs: time=... level=INFO msg="ro.Next" value=user login
// Logs: time=... level=INFO msg="ro.Next" value=data processing
// Logs: time=... level=INFO msg="ro.Next" value=task completed
// Next: user login
// Next: data processing
// Next: task completed
// Completed
```