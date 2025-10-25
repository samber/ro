---
name: Quote
slug: quote
sourceRef: plugins/strconv/operator.go#L205
type: plugin
category: strconv
signatures:
  - "func Quote()"
playUrl: https://go.dev/play/p/H8rujLROgrd
variantHelpers:
  - plugin#strconv#quote
similarHelpers:
  - plugin#strconv#quoterune
  - plugin#strconv#unquote
position: 13
---

Converts strings to Go string literals using strconv.Quote.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, string](
    ro.Just("hello", "world\n", "test\t\"quote\"", "path\\to\\file"),
    rostrconv.Quote(),
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

// Next: "hello"
// Next: "world\n"
// Next: "test\t\"quote\""
// Next: "path\\to\\file"
// Completed
```