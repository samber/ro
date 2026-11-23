---
name: PhoneNumber
slug: phonenumber
sourceRef: plugins/faker/operator.go#L134
type: plugin
category: faker
signatures:
  - "func PhoneNumber(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#phonenumber
similarHelpers:
  []
position: 80
---

Emits `count` random phone numbers, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.PhoneNumber(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 555-010-4477
// Next: ...
// Next: ...
// Completed
```
