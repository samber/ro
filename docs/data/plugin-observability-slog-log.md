---
name: Log
slug: log
sourceRef: plugins/observability/slog/operator.go#L26
type: plugin
category: logger-slog
signatures:
  - "func Log[T any](logger slog.Logger, level slog.Level)"
playUrl: ""
variantHelpers:
  - plugin#logger-slog#log
similarHelpers:
  - plugin#logger-slog#logwithnotification
position: 0
---

Logs with structured logging.

```go
import (
    "log/slog"
    "os"

    "github.com/samber/ro"
    roslog "github.com/samber/ro/plugins/observability/slog"
)

logger := slog.New(slog.NewTextHandler(os.Stdout, nil))

obs := ro.Pipe[string, string](
    ro.Just("operation 1", "operation 2"),
    roslog.Log[string](logger, slog.LevelInfo),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Logs: time=... level=INFO msg="operation 1"
// Logs: time=... level=INFO msg="operation 2"
// Next: operation 1
// Next: operation 2
// Completed
```