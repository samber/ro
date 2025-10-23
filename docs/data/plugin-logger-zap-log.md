---
name: Log
slug: log
sourceRef: plugins/observability/zap/operator.go#L27
type: plugin
category: logger-zap
signatures:
  - "func Log[T any](logger *zap.Logger, level zapcore.Level)"
playUrl:
variantHelpers:
  - plugin#logger-zap#log
similarHelpers: []
position: 0
---

Logs all observable notifications (Next, Error, Complete) using zap logger with formatted messages.

```go
import (
    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugin/logger-zap"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

logger, _ := zap.NewDevelopment()
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    rozap.Log[int](logger, zapcore.InfoLevel),
)

sub := obs.Subscribe(ro.NoopObserver[int]())
defer sub.Unsubscribe()

// zap logs with formatted messages
// INFO	ro.Next: 1
// INFO	ro.Next: 2
// INFO	ro.Next: 3
// INFO	ro.Next: 4
// INFO	ro.Next: 5
// INFO	ro.Complete
```
