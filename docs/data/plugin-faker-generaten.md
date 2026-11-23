---
name: GenerateN
slug: generaten
sourceRef: plugins/faker/operator.go#L82
type: plugin
category: faker
signatures:
  - "func GenerateN[T any](count int64, generator func() (T, error))"
playUrl: ""
variantHelpers:
  - plugin#faker#generaten
similarHelpers:
  - plugin#faker#generate
  - plugin#faker#fake
position: 10
---

Emits `count` values returned by a generator, then completes. A negative count or a nil generator panics. A generator error is forwarded as an error notification and ends the stream.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.GenerateN(3, func() (string, error) {
    return "hello", nil
})

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello
// Next: ...
// Next: ...
// Completed
```
