---
name: Unquote
slug: unquote
sourceRef: plugins/strconv/operator.go#L229
type: plugin
category: strconv
signatures:
  - "func Unquote()"
playUrl: https://go.dev/play/p/ljSAvIMOzmh
variantHelpers:
  - plugin#strconv#unquote
similarHelpers:
  - plugin#strconv#quote
  - plugin#strconv#quoterune
position: 15
---

Converts Go string literals back to strings using strconv.Unquote.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[string, string](
    ro.Just(`"hello"`, `"world\n"`, `"test\t\"quote\""`, `"invalid\quote"`),
    rostrconv.Unquote(),
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

// Next: hello
// Next: world
// Next: test	"quote"
// Error: invalid syntax
```