---
name: ReplaceAllString
slug: replaceallstring
sourceRef: plugins/regexp/operator.go#L101
type: plugin
category: regexp
signatures:
  - "func ReplaceAllString[T ~string](pattern *regexp.Regexp, repl T)"
playUrl: ""
variantHelpers:
  - plugin#regexp#replaceallstring
similarHelpers:
  - plugin#regexp#replaceall
  - plugin#regexp#find
  - plugin#regexp#findstring
position: 120
---

Replaces all matches of the pattern in the string with the replacement.

```go
import (
    "regexp"

    "github.com/samber/ro"
    roregexp "github.com/samber/ro/plugins/regexp"
)

pattern := regexp.MustCompile(`\bworld\b`)
obs := ro.Pipe[string, string](
    ro.Just("hello world", "world peace", "new world order"),
    roregexp.ReplaceAllString[string](pattern, "universe"),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: hello universe
// Next: universe peace
// Next: new universe order
// Completed
```