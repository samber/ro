---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/slog/operator.go#L40
type: plugin
category: logger-slog
signatures:
  - "func LogWithNotification[T any](logger slog.Logger, level slog.Level)"
playUrl: https://go.dev/play/p/M8N_igKHWGV
variantHelpers:
  - plugin#logger-slog#logwithnotification
similarHelpers:
  - plugin#logger-slog#log
position: 10
---

Logs each notification (Next, Error, Complete) emitted by the source Observable using structured logging with slog.

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
    roslog.LogWithNotification[string](logger, slog.LevelInfo),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// time=2009-11-10T23:00:00.000Z level=INFO msg=ro.Next value="operation 1"
// Next: operation 1
// time=2009-11-10T23:00:00.000Z level=INFO msg=ro.Next value="operation 2"
// Next: operation 2
// time=2009-11-10T23:00:00.000Z level=INFO msg=ro.Complete
// Completed
```
