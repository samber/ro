---
name: FindStringSubmatch
slug: findstringsubmatch
sourceRef: plugins/regexp/operator.go#L45
type: plugin
category: regexp
signatures:
  - "func FindStringSubmatch[T ~string](pattern *regexp.Regexp)"
playUrl: https://go.dev/play/p/1YLeGzHmWJS
variantHelpers:
  - plugin#regexp#findstringsubmatch
similarHelpers:
  - plugin#regexp#findstring
  - plugin#regexp#findsubmatch
  - plugin#regexp#findallstringsubmatch
position: 50
---

Finds the first submatch of the pattern in the string.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`(\w+)\s+(\w+)`)
obs := ro.Pipe[string, []string](
    ro.Just("hello world", "foo bar", "test"),
    roregexp.FindStringSubmatch[string](pattern),
)

sub := obs.Subscribe(ro.PrintObserver[[]string]())
defer sub.Unsubscribe()

// Next: [hello world hello world]
// Next: [foo bar foo bar]
// Next: []
// Completed
```