---
name: LogWithNotification
slug: logwithnotification
sourceRef: plugins/observability/zerolog/operator.go#L39
type: plugin
category: logger-zerolog
signatures:
  - "func LogWithNotification[T any](logger *zerolog.Logger, level zerolog.Level)"
playUrl:
variantHelpers:
  - plugin#logger-zerolog#logwithnotification
similarHelpers: []
position: 10
---

Logs all observable notifications using zerolog logger with structured notification data.

```go
import (
  "github.com/samber/ro"
  rozerolog "github.com/samber/ro/plugin/logger-zerolog"
  "github.com/rs/zerolog"
)

logger := zerolog.New(os.Stdout).With().Logger()
obs := ro.Pipe[string, string](
    ro.Just("hello", "world", "golang"),
    rozerolog.LogWithNotification[string](&logger, zerolog.DebugLevel),
)

sub := obs.Subscribe(ro.NoopObserver[string]())
defer sub.Unsubscribe()

//  JSON logs with structured data
// {"level":"debug","value":"hello","message":"ro.Next"}
// {"level":"debug","value":"world","message":"ro.Next"}
// {"level":"debug","value":"golang","message":"ro.Next"}
// {"level":"debug","message":"ro.Complete"}
```
