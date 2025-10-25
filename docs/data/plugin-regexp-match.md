---
name: Match
slug: match
sourceRef: plugins/regexp/operator.go#L80
type: plugin
category: regexp
signatures:
  - "func Match[T ~[]byte](pattern *regexp.Regexp)"
playUrl: https://go.dev/play/p/BNIg4nj8eCf
variantHelpers:
  - plugin#regexp#match
similarHelpers:
  - plugin#regexp#matchstring
  - plugin#regexp#filtermatch
  - plugin#regexp#find
position: 100
---

Checks if the pattern matches the byte slice.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`hello`)
obs := ro.Pipe[[]byte, bool](
    ro.Just(
        []byte("hello world"),
        []byte("goodbye world"),
        []byte("hello again"),
    ),
    roregexp.Match[[]byte](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Next: false
// Next: true
// Completed
```