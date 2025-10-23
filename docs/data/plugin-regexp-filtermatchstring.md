---
name: FilterMatchString
slug: filtermatchstring
sourceRef: plugins/regexp/operator.go#L115
type: plugin
category: regexp
signatures:
  - "func FilterMatchString[T ~string](pattern *regexp.Regexp)"
playUrl: ""
variantHelpers:
  - plugin#regexp#filtermatchstring
similarHelpers:
  - plugin#regexp#filtermatch
  - plugin#regexp#matchstring
  - plugin#regexp#match
position: 140
---

Filters strings that match pattern.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`hello`)
obs := ro.Pipe[string, string](
    ro.Just("hello world", "goodbye world", "hello again", "no match"),
    roregexp.FilterMatchString[string](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello world
// Next: hello again
// Completed
```