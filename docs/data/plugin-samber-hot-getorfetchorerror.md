---
name: GetOrFetchOrError
slug: getorfetchorerror
sourceRef: plugins/samber/hot/operator_hot.go#L81
type: plugin
category: samber-hot
signatures:
  - "func GetOrFetchOrError[K comparable, V any](cache *hot.HotCache[K, V])"
playUrl: ""
variantHelpers:
  - plugin#samber-hot#getorfetchorerror
similarHelpers:
  - plugin#samber-hot#getorfetch
  - plugin#samber-hot#getorfetorskip
position: 10
---

Gets values from hot cache or returns error if not found.

```go
import (
    "fmt"
    "time"

    "github.com/samber/ro"
    rohot "github.com/samber/ro/plugins/samber/hot"
    "github.com/redis/go-redis/v9"
)

cache := hot.NewHotCache[string, string](hot.HotCacheOptions[string, string]{
    TTL: 5 * time.Minute,
})

cache.Set("key1", "value1")

obs := ro.Pipe[string, string](
    ro.Just("key1", "key2", "key3"),
    rohot.GetOrFetchOrError[string, string](cache),
)

sub := obs.Subscribe(
    ro.NewObserver[string](
        func(value string) {
            fmt.Printf("Next: %s\n", value)
        },
        func(err error) {
            fmt.Printf("Error: %v\n", err)
        },
        func() {
            fmt.Println("Completed")
        },
    ),
)
defer sub.Unsubscribe()

// Next: value1
// Error: rohot.GetOrFetchOrError: not found
// Error: rohot.GetOrFetchOrError: not found
// Completed
```