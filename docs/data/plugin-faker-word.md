---
name: Word
slug: word
sourceRef: plugins/faker/operator.go#L159
type: plugin
category: faker
signatures:
  - "func Word(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#word
similarHelpers:
  - plugin#faker#sentence
  - plugin#faker#paragraph
position: 130
---

Emits `count` random lorem ipsum words, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Word(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: voluptas
// Next: ...
// Next: ...
// Completed
```
