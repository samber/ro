---
name: Log
slug: log
sourceRef: plugins/observability/zerolog/operator.go#L27
type: plugin
category: logger-zerolog
signatures:
  - "func Log[T any](logger *zerolog.Logger, level zerolog.Level)"
playUrl: https://go.dev/play/p/wJm9Mw1hkhB
variantHelpers:
  - plugin#logger-zerolog#log
similarHelpers: []
position: 0
---

Logs all observable notifications (Next, Error, Complete) using zerolog logger with formatted messages.

```go
import (
    "github.com/samber/ro"
    rozerolog "github.com/samber/ro/plugins/observability/zerolog"
    "github.com/rs/zerolog"
)

logger := zerolog.New(os.Stdout).With().Logger()
obs := ro.Pipe[int, int](
    ro.Just(1, 2, 3, 4, 5),
    rozerolog.Log[int](&logger, zerolog.InfoLevel),
)

sub := obs.Subscribe(ro.NoopObserver[int]())
defer sub.Unsubscribe()

// JSON logs with formatted messages
// {"level":"info","message":"ro.Next: 1"}
// {"level":"info","message":"ro.Next: 2"}
// {"level":"info","message":"ro.Next: 3"}
// {"level":"info","message":"ro.Next: 4"}
// {"level":"info","message":"ro.Next: 5"}
// {"level":"info","message":"ro.Complete"}
```
