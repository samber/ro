---
name: NewRateLimiter
slug: newratelimiter
sourceRef: plugins/ratelimit/native/operator.go#L24
type: plugin
category: ratelimit-native
signatures:
  - "func NewRateLimiter[T any](count int64, interval time.Duration, keyGetter func(T) string)"
playUrl: ""
variantHelpers:
  - plugin#ratelimit-native#newratelimiter
similarHelpers:
  - plugin#ratelimit-ulule#newratelimiter
position: 0
---

Creates a rate limiter using native implementation.

```go
import (
    "time"

    "github.com/samber/ro"
    roratelimit "github.com/samber/ro/plugins/ratelimit/native"
)

obs := ro.Pipe[string, string](
    ro.Just("user1", "user1", "user2", "user1", "user2"),
    roratelimit.NewRateLimiter[string](2, time.Second, func(s string) string { return s }),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: user1
// Next: user1
// Next: user2
// Next: user2
// (user1 may be dropped due to rate limit)
// Completed
```