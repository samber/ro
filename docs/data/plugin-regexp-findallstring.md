---
name: FindAllString
slug: findallstring
sourceRef: plugins/regexp/operator.go#L56
type: plugin
category: regexp
signatures:
  - "func FindAllString[T ~string](pattern *regexp.Regexp, n int)"
playUrl: https://go.dev/play/p/8nVqsO48nmU
variantHelpers:
  - plugin#regexp#findallstring
similarHelpers:
  - plugin#regexp#findall
  - plugin#regexp#findstring
position: 5
---

Finds all matches of a regex pattern in strings.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`\d+`)
obs := ro.Pipe[string, []string](
    ro.Just("abc123def456", "789ghi012"),
    roregexp.FindAllString[string](pattern, -1), // -1 for unlimited matches
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Next: [123 456]
// Next: [789 012]
// Completed
```