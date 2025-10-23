---
name: FindAllStringSubmatch
slug: findallstringsubmatch
sourceRef: plugins/regexp/operator.go#L70
type: plugin
category: regexp
signatures:
  - "func FindAllStringSubmatch[T ~string](pattern *regexp.Regexp, n int)"
playUrl: ""
variantHelpers:
  - plugin#regexp#findallstringsubmatch
similarHelpers:
  - plugin#regexp#findallsubmatch
  - plugin#regexp#findstringsubmatch
position: 7
---

Finds all submatches of a regex pattern in strings.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`(\d+)-([a-z]+)`)
obs := ro.Pipe[string, [][]string](
    ro.Just("123-abc 456-def", "789-ghi"),
    roregexp.FindAllStringSubmatch[string](pattern, -1), // -1 for unlimited matches
)

sub := obs.Subscribe(ro.PrintObserver[[][]string]())
defer sub.Unsubscribe()

// Next: [[123-abc 123 abc] [456-def 456 def]]
// Next: [[789-ghi 789 ghi]]
// Completed
```