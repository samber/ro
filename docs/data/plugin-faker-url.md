---
name: URL
slug: url
sourceRef: plugins/faker/operator.go#L154
type: plugin
category: faker
signatures:
  - "func URL(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#url
similarHelpers:
  []
position: 120
---

Emits `count` random URLs, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.URL(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: https://www.example.com/path
// Next: ...
// Next: ...
// Completed
```
