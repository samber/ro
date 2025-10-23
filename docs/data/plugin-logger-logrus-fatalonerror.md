---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/logrus/operator.go#L53
type: plugin
category: logger-logrus
signatures:
  - "func FatalOnError[T any](logger *logrus.Logger)"
playUrl: ""
variantHelpers:
  - plugin#logger-logrus#fatalonerror
similarHelpers:
  - plugin#logger-logrus#log
  - plugin#logger-logrus#logwithnotification
position: 20
---

Fatal logs errors using logrus and terminates the application.

```go
import (
    "fmt"
    "github.com/samber/ro"
    "github.com/sirupsen/logrus"
    rologrus "github.com/samber/ro/plugins/observability/logrus"
)

logger := logrus.New()
logger.SetLevel(logrus.ErrorLevel)

obs := ro.Pipe[string, string](
    ro.Just("success", "error occurred"),
    ro.Map[string, string](func(s string) (string, error) {
        if s == "error occurred" {
            return "", fmt.Errorf("processing failed")
        }
        return s, nil
    }),
    rologrus.FatalOnError[string](logger),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Logs: time=... level=FATAL msg="ro.Error" error=processing failed
// (Application terminates)
```