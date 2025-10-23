---
name: FilterMatch
slug: filtermatch
sourceRef: plugins/regexp/operator.go#L109
type: plugin
category: regexp
signatures:
  - "func FilterMatch[T ~[]byte](pattern *regexp.Regexp)"
playUrl: ""
variantHelpers:
  - plugin#regexp#filtermatch
similarHelpers:
  - plugin#regexp#filtermatchstring
position: 130
---

Filters byte slices that match pattern.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`hello`)
obs := ro.Pipe[[]byte, []byte](
    ro.Just(
        []byte("hello world"),
        []byte("goodbye world"),
        []byte("hello again"),
    ),
    roregexp.FilterMatch[[]byte](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 32 119 111 114 108 100]
// Next: [104 101 108 108 111 32 97 103 97 105 110]
// Completed
```