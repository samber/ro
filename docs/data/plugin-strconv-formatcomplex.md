---
name: FormatComplex
slug: formatcomplex
sourceRef: plugins/strconv/operator.go#L135
type: plugin
category: strconv
signatures:
  - "func FormatComplex(mt byte, prec, bitSize int)"
playUrl: https://go.dev/play/p/qT817diKCxy
variantHelpers:
  - plugin#strconv#formatcomplex
similarHelpers:
  - plugin#strconv#formatfloat
  - plugin#strconv#formatint
position: 9
---

Converts complex128 values to strings with specified format, precision, and bit size.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[complex128, string](
    ro.Just(3+4i, 1+2i, 0.5+1.25i),
    rostrconv.FormatComplex('f', 2, 128), // Fixed-point, 2 decimal places
)

sub := obs.Subscribe(ro.NewObserver(
    func(s string) {
        fmt.Printf("Next: %s\n", s)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: (3.00+4.00i)
// Next: (1.00+2.00i)
// Next: (0.50+1.25i)
// Completed
```