---
name: Email
slug: email
sourceRef: plugins/faker/operator.go#L124
type: plugin
category: faker
signatures:
  - "func Email(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#email
similarHelpers:
  []
position: 60
---

Emits `count` random email addresses, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Email(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: alice.johnson@example.org
// Next: ...
// Next: ...
// Completed
```
