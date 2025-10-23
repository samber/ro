---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/zap/operator.go#L41
type: plugin
category: logger-zap
signatures:
  - "func LogWithNotification[T any](logger *zap.Logger, level zapcore.Level)"
playUrl:
variantHelpers:
  - plugin#logger-zap#logwithnotification
similarHelpers: []
position: 10
---

Logs all observable notifications using zap logger with structured notification data.

```go
import (
    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugin/logger-zap"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

logger, _ := zap.NewDevelopment()
obs := ro.Pipe[string, string](
    ro.Just("hello", "world", "golang"),
    rozap.LogWithNotification[string](logger, zapcore.DebugLevel),
)

sub := obs.Subscribe(ro.NoopObserver[string]())
defer sub.Unsubscribe()

//  zap logs with structured data
// DEBUG	ro.Next	{"value": "hello"}
// DEBUG	ro.Next	{"value": "world"}
// DEBUG	ro.Next	{"value": "golang"}
// DEBUG	ro.Complete
```
