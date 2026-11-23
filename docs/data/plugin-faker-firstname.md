---
name: FirstName
slug: firstname
sourceRef: plugins/faker/operator.go#L114
type: plugin
category: faker
signatures:
  - "func FirstName(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#firstname
similarHelpers:
  - plugin#faker#name
  - plugin#faker#lastname
position: 40
---

Emits `count` random first names, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.FirstName(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Alice
// Next: ...
// Next: ...
// Completed
```
