---
name: LastName
slug: lastname
sourceRef: plugins/faker/operator.go#L119
type: plugin
category: faker
signatures:
  - "func LastName(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#lastname
similarHelpers:
  - plugin#faker#name
  - plugin#faker#firstname
position: 50
---

Emits `count` random last names, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.LastName(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: Johnson
// Next: ...
// Next: ...
// Completed
```
