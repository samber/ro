---
name: Name
slug: name
sourceRef: plugins/faker/operator.go#L109
type: plugin
category: faker
signatures:
  - "func Name(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#name
similarHelpers:
  - plugin#faker#firstname
  - plugin#faker#lastname
position: 30
---

Emits `count` random full names, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Name(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Alice Johnson
// Next: ...
// Next: ...
// Completed
```
