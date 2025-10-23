---
name: FatalOnError
slug: fatalonerror
sourceRef: plugins/observability/zerolog/operator.go#L53
type: plugin
category: logger-zerolog
signatures:
  - "func FatalOnError[T any](logger *zerolog.Logger)"
playUrl:
variantHelpers:
  - plugin#logger-zerolog#fatalonerror
similarHelpers: []
position: 20
---

Terminates the program with a fatal error when an observable error notification occurs using zerolog logger.

```go
import (
    "github.com/samber/ro"
    rozerolog "github.com/samber/ro/plugin/logger-zerolog"
    "github.com/rs/zerolog"
)

logger := zerolog.New(os.Stdout).With().Logger()
obs := ro.Pipe[string, string](
    ro.Throw[string](errors.New("critical error")),
    rozerolog.FatalOnError[string](&logger),
)

sub := obs.Subscribe(ro.NoopObserver[string]())
defer sub.Unsubscribe()

//  program terminates with fatal JSON log
// {"level":"fatal","error":"critical error","message":"ro.Error"}
```
