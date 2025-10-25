---
name: FindSubmatch
slug: findsubmatch
sourceRef: plugins/regexp/operator.go#L38
type: plugin
category: regexp
signatures:
  - "func FindSubmatch[T ~[]byte](pattern *regexp.Regexp)"
playUrl: https://go.dev/play/p/nCs2PqFa8kE
variantHelpers:
  - plugin#regexp#findsubmatch
similarHelpers:
  - plugin#regexp#find
  - plugin#regexp#findstringsubmatch
  - plugin#regexp#findallsubmatch
position: 40
---

Finds the first submatch of the pattern in the byte slice.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`(\w+)\s+(\w+)`)
obs := ro.Pipe[[]byte, [][]byte](
    ro.Just(
        []byte("hello world"),
        []byte("foo bar"),
        []byte("test"),
    ),
    roregexp.FindSubmatch[[]byte](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[[][]byte]())
defer sub.Unsubscribe()

// Next: [[104 101 108 108 111 32 119 111 114 108 100] [104 101 108 108 111] [119 111 114 108 100]]
// Next: [[102 111 111 32 98 97 114] [102 111 111] [98 97 114]]
// Next: []
// Completed
```