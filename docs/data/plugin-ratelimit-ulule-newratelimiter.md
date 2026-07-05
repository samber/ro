---
name: NewRateLimiter
slug: newratelimiter
sourceRef: plugins/ratelimit/ulule/operator.go#L27
type: plugin
category: ratelimit-ulule
signatures:
  - "func NewRateLimiter[T any](limiter *limiter.Limiter, keyGetter func(T) string)"
playUrl: https://go.dev/play/p/V4meCiGc3bx
variantHelpers:
  - plugin#ratelimit-ulule#newratelimiter
similarHelpers:
  - plugin#ratelimit-native#newratelimiter
position: 0
---

Rate limits observable values using ulule/limiter with custom key extraction.

```go
import (
    "fmt"
    "time"

    "github.com/samber/ro"
    roratelimit "github.com/samber/ro/plugins/ratelimit/ulule"
    "github.com/ulule/limiter/v3"
    memory "github.com/ulule/limiter/v3/drivers/store/memory"
)

type Request struct {
    UserID string
    Action string
}

store := memory.NewStore()
lim := limiter.New(store, limiter.Rate{
    Period: time.Hour,
    Limit:  100,
})

obs := ro.Pipe[Request, Request](
    ro.Just(
        Request{UserID: "user1", Action: "login"},
        Request{UserID: "user2", Action: "login"},
        Request{UserID: "user1", Action: "post"},
    ),
    roratelimit.NewRateLimiter(lim, func(r Request) string {
        return r.UserID // Rate limit per user
    }),
)

values, err := ro.Collect(obs)
if err != nil {
    fmt.Printf("Error: %v\n", err)
    return
}

for _, v := range values {
    fmt.Printf("Next: %+v\n", v)
}
fmt.Println("Completed")

// Next: {UserID:user1 Action:login}
// Next: {UserID:user2 Action:login}
// Next: {UserID:user1 Action:post}
// Completed
```