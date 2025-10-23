---
name: GetOrFetchOrSkip
slug: getorfetorskip
sourceRef: plugins/samber/hot/operator_hot.go#L54
type: plugin
category: samber-hot
signatures:
  - "func GetOrFetchOrSkip[K comparable, V any](cache *hot.HotCache[K, V])"
playUrl: ""
variantHelpers:
  - plugin#samber-hot#getorfetorskip
similarHelpers:
  - plugin#samber-hot#getorfetch
  - plugin#samber-hot#getorfetchorerror
position: 20
---

Gets values from hot cache and skips items that are not found.

```go
import (
    "time"

    "github.com/samber/ro"
    rohot "github.com/samber/ro/plugins/samber/hot"
    "github.com/redis/go-redis/v9"
)

cache := hot.NewHotCache[string, string](hot.HotCacheOptions[string, string]{
    TTL: 5 * time.Minute,
})

cache.Set("key1", "value1")
cache.Set("key3", "value3")

obs := ro.Pipe[string, string](
    ro.Just("key1", "key2", "key3", "key4"),
    rohot.GetOrFetchOrSkip[string, string](cache),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: value1
// Next: value3
// Completed
```

Only items found in the cache are emitted, missing items are silently skipped.