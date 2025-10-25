---
name: Atoi
slug: atoi
sourceRef: plugins/strconv/operator.go#L31
type: plugin
category: strconv
signatures:
  - "func Atoi[T ~string]()"
playUrl: https://go.dev/play/p/eaJ8rivjFzR
variantHelpers:
  - plugin#strconv#atoi
similarHelpers:
  - plugin#strconv#parseint
  - plugin#strconv#parseuint
position: 0
---

Converts strings to integers using strconv.Atoi.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, int](
    ro.Just("123", "456", "789", "invalid"),
    rostrconv.Atoi[string](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(i int) {
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

// Next: 123
// Next: 456
// Next: 789
// Error: strconv.Atoi: parsing "invalid": invalid syntax
```