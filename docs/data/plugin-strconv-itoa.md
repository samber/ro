---
name: Itoa
slug: itoa
sourceRef: plugins/strconv/operator.go#L194
type: plugin
category: strconv
signatures:
  - "func Itoa()"
playUrl: ""
variantHelpers:
  - plugin#strconv#itoa
similarHelpers:
  - plugin#strconv#formatbool
  - plugin#strconv#formatint
  - plugin#strconv#formatuint
position: 60
---

Converts integers to strings using strconv.Itoa.

This is equivalent to FormatInt with base 10.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[int, string](
    ro.Just(123, -456, 0, 789),
    rostrconv.Itoa(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 123
// Next: -456
// Next: 0
// Next: 789
// Completed
```