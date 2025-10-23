---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/sentry/operator.go#L50
type: plugin
category: logger-sentry
signatures:
  - "func LogWithNotification[T any](logger *sentry.Hub, level sentry.Level)"
playUrl: ""
variantHelpers:
  - plugin#logger-sentry#log
  - plugin#logger-sentry#logwithnotification
similarHelpers:
  - plugin#logger-sentry#log
position: 10
---

Logs events to Sentry with structured data and notifications.

```go
import (
    "github.com/getsentry/sentry-go"
    "github.com/samber/ro"
    rosentry "github.com/samber/ro/plugins/observability/sentry"
)

hub := sentry.NewHub(sentry.CurrentClient(), sentry.NewScope())
obs := ro.Pipe[string, string](
    ro.Just("user login", "data processing", "error occurred"),
    rosentry.LogWithNotification[string](hub, sentry.LevelInfo),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Sentry events captured with structured data for: user login, data processing, error occurred
// Next: user login
// Next: data processing
// Next: error occurred
// Completed
```