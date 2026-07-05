---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/zap/operator.go#L61
type: plugin
category: logger-zap
signatures:
  - "func FatalOnError[T any](logger *zap.Logger)"
playUrl: https://go.dev/play/p/00E6cS_aAWU
variantHelpers:
  - plugin#logger-zap#fatalonerror
similarHelpers: []
position: 20
---

Terminates the program with a fatal error when an observable error notification occurs using zap logger.

```go
import (
    "errors"

    "github.com/samber/ro"
    rozap "github.com/samber/ro/plugins/observability/zap"
    "go.uber.org/zap"
)

logger := zap.NewExample()

sub := ro.Pipe[string, string](
    ro.Throw[string](errors.New("critical error")),
    rozap.FatalOnError[string](logger),
).Subscribe(ro.NoopObserver[string]())
defer sub.Unsubscribe()

// {"level":"fatal","msg":"ro.Error","error":"critical error"}
// exit status 1
```
