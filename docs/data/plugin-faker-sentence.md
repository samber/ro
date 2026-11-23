---
name: Sentence
slug: sentence
sourceRef: plugins/faker/operator.go#L164
type: plugin
category: faker
signatures:
  - "func Sentence(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#sentence
similarHelpers:
  - plugin#faker#word
  - plugin#faker#paragraph
position: 140
---

Emits `count` random lorem ipsum sentences, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Sentence(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Quia est ut dolores.
// Next: ...
// Next: ...
// Completed
```
