---
name: Username
slug: username
sourceRef: plugins/faker/operator.go#L129
type: plugin
category: faker
signatures:
  - "func Username(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#username
similarHelpers:
  []
position: 70
---

Emits `count` random user names, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Username(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: alicejohnson
// Next: ...
// Next: ...
// Completed
```
