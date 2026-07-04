---
name: FatalOnErrorWithPrefix
slug: fatalonerrorwithprefix
sourceRef: plugins/observability/log/operator.go#L59
type: plugin
category: logger-log
signatures:
  - "func FatalOnErrorWithPrefix[T any](prefix string)"
playUrl: https://go.dev/play/p/9E0dtrSJrxE
variantHelpers:
  - plugin#logger-log#fatalonerrorwithprefix
similarHelpers:
  - plugin#logger-log#logwithprefix
  - plugin#logger-log#fatalonerror
position: 3
---

Terminates the application on error with prefixed logging.

```go
import (
    "errors"
    "fmt"
    "log"
    "os"

    "github.com/samber/ro"
    rolog "github.com/samber/ro/plugins/observability/log"
)

log.SetOutput(os.Stdout)
log.SetFlags(0)

obs := ro.Pipe[int, int](
    ro.Concat[int](ro.Just(1, 2), ro.Throw[int](errors.New("fatal error"))),
    rolog.FatalOnErrorWithPrefix[int]("Critical"),
)

sub := obs.Subscribe(ro.NewObserver[int](
    func(v int) { fmt.Printf("Next: %d\n", v) },
    func(err error) { fmt.Printf("Error: %v\n", err) },
    func() { fmt.Println("Completed") },
))
defer sub.Unsubscribe()

// Next: 1
// Next: 2
// Critical ro.Error: fatal error
// (program exits with code 1)
```