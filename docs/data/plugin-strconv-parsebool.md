---
name: ParseBool
slug: parsebool
sourceRef: plugins/strconv/operator.go#L75
type: plugin
category: strconv
signatures:
  - "func ParseBool[T ~string]()"
playUrl: https://go.dev/play/p/lcXbpr9UIVT
variantHelpers:
  - plugin#strconv#parsebool
similarHelpers:
  - plugin#strconv#atoi
  - plugin#strconv#parseint
  - plugin#strconv#parsefloat
position: 30
---

Converts strings to boolean values using strconv.ParseBool.

Accepts "1", "t", "T", "true", "TRUE", "True" for true values.
Accepts "0", "f", "F", "false", "FALSE", "False" for false values.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, bool](
    ro.Just("true", "false", "1", "0", "TRUE", "FALSE"),
    rostrconv.ParseBool[string](),
)

sub := obs.Subscribe(ro.PrintObserver[bool]())
defer sub.Unsubscribe()

// Next: true
// Next: false
// Next: true
// Next: false
// Next: true
// Next: false
// Completed
```