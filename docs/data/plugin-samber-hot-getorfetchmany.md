---
name: GetOrFetchMany
slug: getorfetchmany
sourceRef: plugins/samber/hot/operator_hot.go#L110
type: plugin
category: samber-hot
signatures:
  - "func GetOrFetchMany[K comparable, V any](cache *hot.HotCache[K, V])"
playUrl: ""
variantHelpers:
  - plugin#samber-hot#getorfetchmany
similarHelpers:
  - plugin#samber-hot#getorfetch
position: 30
---

Gets multiple values from hot cache by keys.

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
cache.Set("key2", "value2")
cache.Set("key4", "value4")

obs := ro.Pipe[[]string, map[string]string](
    ro.Just(
        []string{"key1", "key2", "key3"},
        []string{"key2", "key4", "key5"},
    ),
    rohot.GetOrFetchMany[string, string](cache),
)

sub := obs.Subscribe(ro.PrintObserver[map[string]string]())
defer sub.Unsubscribe()

// Next: map[key1:value1 key2:value2]
// Next: map[key2:value2 key4:value4]
// Completed
```

Returns a map containing only the keys that were found in the cache.