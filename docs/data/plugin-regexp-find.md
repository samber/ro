---
name: Find
slug: find
sourceRef: plugins/regexp/operator.go#L25
type: plugin
category: regexp
signatures:
  - "func Find[T ~[]byte](pattern *regexp.Regexp)"
playUrl: ""
variantHelpers:
  - plugin#regexp#find
similarHelpers:
  - plugin#regexp#findstring
  - plugin#regexp#findall
position: 0
---

Finds the first match of a regex pattern in byte slices.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`\d+`)
obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("abc123def"), []byte("no numbers here")),
    roregexp.Find[[]byte](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [49 50 51]
// Next: []
// Completed
```