---
name: NewPrompt
slug: newprompt
sourceRef: plugins/io/source.go#L95
type: plugin
category: stdio
signatures:
  - "func NewPrompt(prompt string)"
playUrl: ""
variantHelpers:
  - plugin#io#newprompt
similarHelpers:
  - plugin#io#newstdreader
  - plugin#io#newstdreaderline
position: 4
---

Creates an observable that prompts the user for input.

```go
import (
    "github.com/samber/ro"
    rostdio "github.com/samber/ro/plugins/stdio"
)

obs := rostdio.NewPrompt("Enter your name: ")

sub := obs.Subscribe(ro.NewObserver(
    func(input []byte) {
        println("You entered:", string(input))
    },
    func(err error) {
        println("Error:", err.Error())
    },
    func() {
        println("Input completed")
    },
))
defer sub.Unsubscribe()

// (User sees prompt: "Enter your name: ")
// User types: "Alice"
// You entered: Alice
```