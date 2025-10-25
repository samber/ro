---
name: ReplaceAll
slug: replaceall
sourceRef: plugins/regexp/operator.go#L95
type: plugin
category: regexp
signatures:
  - "func ReplaceAll[T ~[]byte](pattern *regexp.Regexp, repl T)"
playUrl: https://go.dev/play/p/fNTOZi-6YtQ
variantHelpers:
  - plugin#regexp#replaceall
similarHelpers:
  - plugin#regexp#replaceallstring
position: 115
---

Replaces all matches in byte slice.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`world`)
obs := ro.Pipe[[]byte, []byte](
    ro.Just([]byte("hello world, goodbye world")),
    roregexp.ReplaceAll[[]byte](pattern, []byte("universe")),
)

sub := obs.Subscribe(ro.PrintObserver[[]byte]())
defer sub.Unsubscribe()

// Next: [104 101 108 108 111 32 117 110 105 118 101 114 115 101 44 32 103 111 111 100 98 121 101 32 117 110 105 118 101 114 115 101]
// Completed
```