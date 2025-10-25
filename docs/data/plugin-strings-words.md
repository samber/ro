---
name: Words
slug: words
sourceRef: plugins/strings/operator_words.go#L41
type: plugin
category: strings
signatures:
  - "func Words[T ~string]()"
playUrl: https://go.dev/play/p/feiFLt_7lM_0
variantHelpers:
  - plugin#strings#words
similarHelpers:
  - plugin#bytes#words
position: 50
---

Splits string into words.

```go
import (
    "fmt"
    "github.com/samber/ro"
    rostrings "github.com/samber/ro/plugins/strings"
)

obs := ro.Pipe[string, []string](
    ro.Just("hello world from go"),
    rostrings.Words[string](),
)

sub := obs.Subscribe(ro.NewObserver(
    func(words []string) {
        fmt.Printf("Next: %v\n", words)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: [hello world from go]
// Completed
```