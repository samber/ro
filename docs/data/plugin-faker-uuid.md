---
name: UUID
slug: uuid
sourceRef: plugins/faker/operator.go#L139
type: plugin
category: faker
signatures:
  - "func UUID(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#uuid
similarHelpers:
  []
position: 90
---

Emits `count` random hyphenated UUIDs, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.UUID(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 2b7a5e8e-61a1-4a52-9d3c-0d6f8d2f5c11
// Next: ...
// Next: ...
// Completed
```
