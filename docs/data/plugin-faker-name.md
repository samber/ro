---
name: Name
slug: name
sourceRef: plugins/faker/operator.go#L55
type: plugin
category: faker
signatures:
  - "func Name(count int)"
variantHelpers:
  - plugin#faker#name
similarHelpers:
  - plugin#faker#firstname
  - plugin#faker#lastname
position: 1
---

Creates an Observable that emits `count` random full names.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Name(3)

sub := obs.Subscribe(ro.NewObserver(
    func(name string) {
        fmt.Printf("Next: %s\n", name)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: John Smith
// Next: Jane Doe
// Next: Bob Johnson
// Completed
```
