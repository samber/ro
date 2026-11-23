---
name: IPv4
slug: ipv4
sourceRef: plugins/faker/operator.go#L144
type: plugin
category: faker
signatures:
  - "func IPv4(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#ipv4
similarHelpers:
  - plugin#faker#ipv6
position: 100
---

Emits `count` random IPv4 addresses, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.IPv4(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 192.168.12.7
// Next: ...
// Next: ...
// Completed
```
