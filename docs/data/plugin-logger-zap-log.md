---
name: Log
slug: log
sourceRef: plugins/observability/zap/operator.go#L29
type: plugin
category: logger-zap
signatures:
  - "func Log[T any](logger *zap.Logger, level zapcore.Level)"
playUrl: https://go.dev/play/p/3kWjeZo4ciK
variantHelpers:
  - plugin#logger-zap#log
similarHelpers: []
position: 0
---

Logs all observable notifications (Next, Error, Complete) using zap logger with formatted messages.

```go
import (
    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugins/observability/zap"
    "go.uber.org/zap"
    "go.uber.org/zap/zapcore"
)

logger := zap.NewExample()
defer logger.Sync()

_, _ = ro.Collect(ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    rozap.Log[int](logger, zapcore.InfoLevel),
))

// {"level":"info","msg":"ro.Next: 1"}
// {"level":"info","msg":"ro.Next: 2"}
// {"level":"info","msg":"ro.Next: 3"}
// {"level":"info","msg":"ro.Next: 4"}
// {"level":"info","msg":"ro.Next: 5"}
// {"level":"info","msg":"ro.Complete"}
```
