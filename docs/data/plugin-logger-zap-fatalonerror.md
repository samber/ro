---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/zap/operator.go#L55
type: plugin
category: logger-zap
signatures:
  - "func FatalOnError[T any](logger *zap.Logger)"
playUrl:
variantHelpers:
  - plugin#logger-zap#fatalonerror
similarHelpers: []
position: 20
---

Terminates the program with a fatal error when an observable error notification occurs using zap logger.

```go
import (
    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugin/logger-zap"
    "go.uber.org/zap"
)

logger, _ := zap.NewDevelopment()
obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("critical error")),
    rozap.FatalOnError[string](logger),
)

sub := obs.Subscribe(ro.NoopObserver[string]())
defer sub.Unsubscribe()

//  program terminates with fatal log
// FATAL	ro.Error	{"error": "critical error"}
```
