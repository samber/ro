---
name: FormatBool
slug: formatbool
sourceRef: plugins/strconv/operator.go#L116
type: plugin
category: strconv
signatures:
  - "func FormatBool()"
playUrl: https://go.dev/play/p/BWwZ2oDThAK
variantHelpers:
  - plugin#strconv#formatbool
similarHelpers:
  - plugin#strconv#itoa
  - plugin#strconv#formatint
  - plugin#strconv#formatfloat
position: 50
---

Converts boolean values to strings using strconv.FormatBool.

Returns "true" for true values and "false" for false values.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[bool, string](
    ro.Just(true, false, true, false),
    rostrconv.FormatBool(),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: true
// Next: false
// Next: true
// Next: false
// Completed
```