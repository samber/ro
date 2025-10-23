---
name: FindAllSubmatch
slug: findallsubmatch
sourceRef: plugins/regexp/operator.go#L63
type: plugin
category: regexp
signatures:
  - "func FindAllSubmatch[T ~[]byte](pattern *regexp.Regexp, n int)"
playUrl: ""
variantHelpers:
  - plugin#regexp#findallsubmatch
similarHelpers:
  - plugin#regexp#findsubmatch
  - plugin#regexp#findallstringsubmatch
position: 6
---

Finds all submatches of a regex pattern in byte slices.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`(\d+)-([a-z]+)`)
obs := ro.Pipe[[]byte, [][][]byte](
    ro.Just([]byte("123-abc 456-def"), []byte("789-ghi")),
    roregexp.FindAllSubmatch[[]byte](pattern, -1), // -1 for unlimited matches
)

sub := obs.Subscribe(ro.PrintObserver[[][][]byte]())
defer sub.Unsubscribe()

// Next: [[[49 50 51] [97 98 99]] [[52 53 54] [100 101 102]]]
// Next: [[[55 56 57] [103 104 105]]]
// Completed
```