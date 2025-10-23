---
name: GetOrFetch
slug: getorfetch
sourceRef: plugins/samber/hot/operator_hot.go#L29
type: plugin
category: samber-hot
signatures:
  - "func GetOrFetch[K comparable, V any](cache *hot.HotCache[K, V])"
playUrl: ""
variantHelpers:
  - plugin#samber-hot#getorfetch
similarHelpers:
  - plugin#samber-hot#getorfetchorskip
  - plugin#samber-hot#getorfetchorerror
position: 0
---

Gets from cache or fetches if not present.

```go
import (
    hot "github.com/samber/go-hot"
    "github.com/samber/ro"
    rohot "github.com/samber/ro/plugins/samber/hot"
)

cache := hot.New[string, string]()
cache.Set("key1", "value1")

obs := ro.Pipe[string, lo.Tuple2[string, bool]](
    ro.Just("key1", "key2"),
    rohot.GetOrFetch[string, string](cache),
)

sub := obs.Subscribe(ro.PrintObserver[lo.Tuple2[string, bool]]())
defer sub.Unsubscribe()

// Next: {Value: value1, Exists: true}
// Next: {Value: <empty>, Exists: false}
// Completed
```