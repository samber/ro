---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/logrus/operator.go#L43
type: plugin
category: logger-logrus
signatures:
  - "func LogWithNotification[T any](logger *logrus.Logger, level logrus.Level)"
playUrl: https://go.dev/play/p/jaNisbwqa4G
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
    "fmt"
    "os"

    "github.com/samber/ro"
    rologrus "github.com/samber/ro/plugins/observability/logrus"
    "github.com/sirupsen/logrus"
)

logger := logrus.New()
logger.SetOutput(os.Stdout)
logger.SetLevel(logrus.InfoLevel)
logger.SetFormatter(&logrus.TextFormatter{
    DisableColors:    true,
    DisableTimestamp: true,
})

values, err := ro.Collect(
    ro.Pipe1(
        ro.Just("user login", "data processing", "task completed"),
        rologrus.LogWithNotification[string](logger, logrus.InfoLevel),
    ),
)

for _, v := range values {
    fmt.Printf("Next: %s\n", v)
}
if err != nil {
    fmt.Printf("Error: %v\n", err)
} else {
    fmt.Println("Completed")
}

// level=info msg=ro.Next value="user login"
// level=info msg=ro.Next value="data processing"
// level=info msg=ro.Next value="task completed"
// level=info msg=ro.Complete
// Next: user login
// Next: data processing
// Next: task completed
// Completed
```