---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/zap/operator.go#L45
type: plugin
category: logger-zap
signatures:
  - "func LogWithNotification[T any](logger *zap.Logger, level zapcore.Level)"
playUrl: https://go.dev/play/p/XXS2joeg3JN
variantHelpers:
  - plugin#logger-zap#logwithnotification
similarHelpers: []
position: 10
---

Logs all observable notifications using zap logger with structured notification data.

```go
import (
    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugins/observability/zap"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

logger := zap.NewExample()
defer logger.Sync()

_, _ = ro.Collect(ro.Pipe[string, string](
    ro.Just("hello", "world", "golang"),
    rozap.LogWithNotification[string](logger, zapcore.DebugLevel),
))

// {"level":"debug","msg":"ro.Next","value":"hello"}
// {"level":"debug","msg":"ro.Next","value":"world"}
// {"level":"debug","msg":"ro.Next","value":"golang"}
// {"level":"debug","msg":"ro.Complete"}
```
