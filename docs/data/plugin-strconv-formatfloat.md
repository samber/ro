---
name: FormatFloat
slug: formatfloat
sourceRef: plugins/strconv/operator.go#L141
type: plugin
category: strconv
signatures:
  - "func FormatFloat(fmt byte, prec int, bitSize int)"
playUrl: https://go.dev/play/p/GWSPE4Mp-uy
variantHelpers:
  - plugin#strconv#formatfloat
similarHelpers:
  - plugin#strconv#formatint
  - plugin#strconv#formatuint
  - plugin#strconv#parsefloat
position: 70
---

Converts each float64 emitted by the source Observable to its string representation using the given format and precision.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[float64, string](
    ro.Just(3.14159, 2.71828),
    rostrconv.FormatFloat('f', 2, 64),
)

sub := obs.Subscribe(ro.PrintObserver[string]())
defer sub.Unsubscribe()

// Next: 3.14
// Next: 2.72
// Completed
```
