---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/logrus/operator.go#L53
type: plugin
category: logger-logrus
signatures:
  - "func FatalOnError[T any](logger *logrus.Logger)"
playUrl: https://go.dev/play/p/tO7LBvlXY9J
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
    "os"

    "github.com/samber/ro"
    rologrus "github.com/samber/ro/plugins/observability/logrus"
    "github.com/sirupsen/logrus"
)

logger := logrus.New()
logger.SetOutput(os.Stdout)
logger.SetFormatter(&logrus.TextFormatter{
    DisableColors:    true,
    DisableTimestamp: true,
})
// Override exit so the Playground process doesn't terminate
logger.ExitFunc = func(code int) {
    fmt.Printf("[os.Exit(%d) would be called here]\n", code)
}

_, err := ro.Collect(
    ro.Pipe1(
        ro.Throw[string](fmt.Errorf("something went wrong")),
        rologrus.FatalOnError[string](logger),
    ),
)

if err != nil {
    fmt.Printf("Stream error: %v\n", err)
}

// level=fatal msg=ro.Error error="something went wrong"
// [os.Exit(1) would be called here]
// Stream error: something went wrong
```