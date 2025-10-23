---
name: ParseUint64
slug: parseuint64
sourceRef: plugins/strconv/operator.go#L95
type: plugin
category: strconv
signatures:
  - "func ParseUint64[T ~string](base int, bitSize int)"
playUrl: ""
variantHelpers:
  - plugin#strconv#parseuint64
similarHelpers:
  - plugin#strconv#parseuint
  - plugin#strconv#parseint
position: 5
---

Converts strings to uint64 values with specified base and bit size.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, uint64](
    ro.Just("123", "FF", "1010", "invalid"),
    rostrconv.ParseUint64[string](16, 64), // Parse as hex, 64-bit unsigned
)

sub := obs.Subscribe(ro.NewObserver(
    func(i uint64) {
        fmt.Printf("Next: %d\n", i)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: 291
// Next: 255
// Next: 4112
// Error: strconv.ParseUint: parsing "invalid": invalid syntax
```