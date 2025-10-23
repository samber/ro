---
name: ParseInt
slug: parseint
sourceRef: plugins/strconv/operator.go#L46
type: plugin
category: strconv
signatures:
  - "func ParseInt[T ~string](base int, bitSize int)"
playUrl: ""
variantHelpers:
  - plugin#strconv#parseint
similarHelpers:
  - plugin#strconv#parseuint
  - plugin#strconv#parsefloat
position: 10
---

Parses strings to int64 with base and bit size.

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, int64](
    ro.Just("42", "-10", "ff"),
    rostrconv.ParseInt[string](10, 64),
)

sub := obs.Subscribe(ro.PrintObserver[int64]())
defer sub.Unsubscribe()

// Next: 42
// Next: -10
// Error: strconv.ParseInt: parsing "ff": invalid syntax
```