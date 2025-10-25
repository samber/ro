---
name: FindAll
slug: findall
sourceRef: plugins/regexp/operator.go#L49
type: plugin
category: regexp
signatures:
  - "func FindAll[T ~[]byte](pattern *regexp.Regexp, n int)"
playUrl: https://go.dev/play/p/t04D2kQJq2-
variantHelpers:
  - plugin#regexp#findall
similarHelpers:
  - plugin#regexp#find
  - plugin#regexp#findallstring
position: 4
---

Finds all matches of a regex pattern in byte slices.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`\d+`)
obs := ro.Pipe[[]byte, [][]byte](
    ro.Just([]byte("abc123def456"), []byte("789ghi012")),
    roregexp.FindAll[[]byte](pattern, -1), // -1 for unlimited matches
)

sub := obs.Subscribe(ro.PrintObserver[[][]byte]())
defer sub.Unsubscribe()

// Next: [[49 50 51] [52 53 54]]
// Next: [[55 56 57] [48 49 50]]
// Completed
```