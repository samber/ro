---
name: Log
slug: log
sourceRef: plugins/observability/slog/operator.go#L26
type: plugin
category: logger-slog
signatures:
  - "func Log[T any](logger slog.Logger, level slog.Level)"
playUrl: https://go.dev/play/p/-94jOwZbMtx
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

// time=2009-11-10T23:00:00.000Z level=INFO msg="ro.Next: operation 1"
// Next: operation 1
// time=2009-11-10T23:00:00.000Z level=INFO msg="ro.Next: operation 2"
// Next: operation 2
// time=2009-11-10T23:00:00.000Z level=INFO msg=ro.Complete
// Completed
```