---
name: FindString
slug: findstring
sourceRef: plugins/regexp/operator.go#L32
type: plugin
category: regexp
signatures:
  - "func FindString[T ~string](pattern *regexp.Regexp)"
playUrl: ""
variantHelpers:
  - plugin#regexp#findstring
similarHelpers:
  - plugin#regexp#find
  - plugin#regexp#findallstring
position: 10
---

Finds the first match of a regex pattern in strings.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`\d+`)
obs := ro.Pipe[string, string](
    ro.Just("abc123def", "no numbers here"),
    roregexp.FindString[string](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 123
// Completed
```