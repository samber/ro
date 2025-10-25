---
name: ParseUint
slug: parseuint
sourceRef: plugins/strconv/operator.go#L90
type: plugin
category: strconv
signatures:
  - "func ParseUint[T ~string](base int, bitSize int)"
playUrl: https://go.dev/play/p/IS3cKM9fFFg
variantHelpers:
  - plugin#strconv#parseuint
similarHelpers:
  - plugin#strconv#parseint
  - plugin#strconv#atoi
  - plugin#strconv#parsefloat
position: 40
---

Converts strings to uint64 values with specified base and bit size.

The base parameter determines the number system (e.g., 10 for decimal, 16 for hexadecimal).
The bitSize parameter specifies the integer type size (e.g., 32 for uint32, 64 for uint64).

```go
import (
    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, uint64](
    ro.Just("255", "FF", "11111111", "377"),
    rostrconv.ParseUint[string](16, 64), // Parse as hexadecimal, 64-bit unsigned
)

sub := obs.Subscribe(ro.PrintObserver[uint64]())
defer sub.Unsubscribe()

// Next: 255
// Next: 255
// Next: 255
// Next: 255
// Completed
```

```go
obs := ro.Pipe[string, uint64](
    ro.Just("123", "456", "789"),
    rostrconv.ParseUint[string](10, 64), // Parse as decimal, 64-bit unsigned
)

sub := obs.Subscribe(ro.PrintObserver[uint64]())
defer sub.Unsubscribe()

// Next: 123
// Next: 456
// Next: 789
// Completed
```