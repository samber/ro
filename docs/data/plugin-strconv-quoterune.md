---
name: QuoteRune
slug: quoterune
sourceRef: plugins/strconv/operator.go#L217
type: plugin
category: strconv
signatures:
  - "func QuoteRune()"
playUrl: https://go.dev/play/p/8evZnIhw4k8
variantHelpers:
  - plugin#strconv#quoterune
similarHelpers:
  - plugin#strconv#quote
  - plugin#strconv#unquote
position: 14
---

Converts runes to Go character literals using strconv.QuoteRune.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rostrconv "github.com/samber/ro/plugins/strconv"
)

obs := ro.Pipe[rune, string](
    ro.Just('a', 'b', '\n', '\t', '"'),
    rostrconv.QuoteRune(),
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

// Next: 'a'
// Next: 'b'
// Next: '\n'
// Next: '\t'
// Next: '"'
// Completed
```