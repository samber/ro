---
name: IPv6
slug: ipv6
sourceRef: plugins/faker/operator.go#L149
type: plugin
category: faker
signatures:
  - "func IPv6(count int64, opts ...options.OptionFunc)"
playUrl: ""
variantHelpers:
  - plugin#faker#ipv6
similarHelpers:
  - plugin#faker#ipv4
position: 110
---

Emits `count` random IPv6 addresses, then completes. A negative count panics. Options are forwarded to go-faker.

```go
import (
    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.IPv6(3)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 2001:db8:85a3::8a2e:370:7334
// Next: ...
// Next: ...
// Completed
```
