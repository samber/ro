---
name: MatchString
slug: matchstring
sourceRef: plugins/regexp/operator.go#L87
type: plugin
category: regexp
signatures:
  - "func MatchString[T ~string](pattern *regexp.Regexp)"
playUrl: https://go.dev/play/p/LYdScuJDXfA
variantHelpers:
  - plugin#regexp#matchstring
similarHelpers:
  - plugin#regexp#match
  - plugin#regexp#filtermatchstring
  - plugin#regexp#findstring
position: 110
---

Checks if the pattern matches the string.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`hello`)
obs := ro.Pipe[string, bool](
    ro.Just("hello world", "goodbye world", "hello again"),
    roregexp.MatchString[string](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Next: false
// Next: true
// Completed
```