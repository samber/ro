---
name: NewRateLimiter
slug: newratelimiter
sourceRef: plugins/ratelimit/ulule/operator.go#L25
type: plugin
category: ratelimit-ulule
signatures:
  - "func NewRateLimiter[T any](limiter *limiter.Limiter, keyGetter func(T) string)"
playUrl: ""
variantHelpers:
  - plugin#ratelimit-ulule#newratelimiter
similarHelpers:
  - plugin#ratelimit-native#newratelimiter
position: 0
---

Rate limits observable values using ulule/limiter with custom key extraction.

```go
import (
    "time"

    "github.com/samber/ro"
    roratelimit "github.com/samber/ro/plugins/ratelimit/ulule"
    "github.com/ulule/limiter"
)

rateLimiter, _ := limiter.New(limiter.Rate{
    Period: time.Hour,
    Limit:  100,
})

type Request struct {
    UserID    string
    Action    string
    Timestamp time.Time
}

obs := ro.Pipe[Request, Request](
    ro.Just(
        Request{UserID: "user1", Action: "login"},
        Request{UserID: "user2", Action: "login"},
        Request{UserID: "user1", Action: "post"},
    ),
    roratelimit.NewRateLimiter(rateLimiter, func(r Request) string {
        return r.UserID // Rate limit per user
    }),
)

sub := obs.Subscribe(ro.PrintObserver[Request]())
defer sub.Unsubscribe()

// Next: {UserID: user1, Action: login}
// Next: {UserID: user2, Action: login}
// Next: {UserID: user1, Action: post}
// Completed
```