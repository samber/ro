---
name: Log
slug: log
sourceRef: plugins/observability/logrus/operator.go#L25
type: plugin
category: logger-logrus
signatures:
  - "func Log[T any](logger *logrus.Logger, level logrus.Level)"
playUrl: https://go.dev/play/p/JMou1n2AIXS
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
        ro.Just("message 1", "message 2"),
        rologrus.Log[string](logger, logrus.InfoLevel),
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

// level=info msg="ro.Next: message 1"
// level=info msg="ro.Next: message 2"
// level=info msg=ro.Complete
// Next: message 1
// Next: message 2
// Completed
```