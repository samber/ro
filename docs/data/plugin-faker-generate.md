---
name: Generate
slug: generate
sourceRef: plugins/faker/operator.go#L72
type: plugin
category: faker
signatures:
  - "func Generate[T any](generator func() (T, error))"
playUrl: ""
variantHelpers:
  - plugin#faker#generate
similarHelpers:
  - plugin#faker#generaten
  - plugin#faker#fake
position: 0
---

Emits an infinite sequence of the values returned by a generator, one call per value. Bound it with `ro.Take`. A generator error is forwarded as an error notification and ends the stream.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

counter := 0

obs := ro.Pipe1(
    rofaker.Generate(func() (int, error) {
        counter++
        return counter * counter, nil
    }),
    ro.Take[int](4),
)

sub := obs.Subscribe(ro.PrintObserver[int]())
defer sub.Unsubscribe()

// Next: 1
// Next: 4
// Next: ...
// Next: ...
// Completed
```
