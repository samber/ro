---
name: ParseFloat
slug: parsefloat
sourceRef: plugins/strconv/operator.go#L60
type: plugin
category: strconv
signatures:
  - "func ParseFloat[T ~string](bitSize int)"
playUrl: ""
variantHelpers:
  - plugin#strconv#parsefloat
similarHelpers:
  - plugin#strconv#parseint
  - plugin#strconv#parsebool
position: 20
---

Converts strings to float values with specified bit size.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, float64](
    ro.Just("3.14", "2.718", "1.414", "invalid"),
    rostrconv.ParseFloat[string](64),
)

sub := obs.Subscribe(ro.NewObserver(
    func(f float64) {
        fmt.Printf("Next: %.3f\n", f)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: 3.140
// Next: 2.718
// Next: 1.414
// Error: strconv.ParseFloat: parsing "invalid": invalid syntax
```