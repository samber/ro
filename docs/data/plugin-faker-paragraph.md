---
name: Paragraph
slug: paragraph
sourceRef: plugins/faker/operator.go#L169
type: plugin
category: faker
signatures:
  - "func Paragraph(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#paragraph
similarHelpers:
  - plugin#faker#word
  - plugin#faker#sentence
position: 150
---

Emits `count` random lorem ipsum paragraphs, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Paragraph(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Quia est ut dolores. Et ...
// Next: ...
// Next: ...
// Completed
```
