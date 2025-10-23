---
name: Log
slug: log
sourceRef: plugins/observability/sentry/operator.go#L26
type: plugin
category: logger-sentry
signatures:
  - "func Log[T any](logger *sentry.Hub, level sentry.Level)"
playUrl: ""
variantHelpers:
  - plugin#logger-sentry#log
  - plugin#logger-sentry#logwithnotification
similarHelpers:
  - plugin#logger-sentry#logwithnotification
position: 0
---

Logs events to Sentry.

```go
import (
    "github.com/samber/ro"
    rosentry "github.com/samber/ro/plugins/observability/sentry"
    "github.com/getsentry/sentry-go"
)

hub := sentry.NewHub(sentry.CurrentClient(), sentry.NewScope())
obs := ro.Pipe[string, string](
    ro.Just("user login", "data processing", "error occurred"),
    rosentry.Log[string](hub, sentry.LevelInfo),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Sentry events captured for: user login, data processing, error occurred
// Next: user login
// Next: data processing
// Next: error occurred
// Completed
```