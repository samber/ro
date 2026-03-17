---
name: Email
slug: email
sourceRef: plugins/faker/operator.go#L101
type: plugin
category: faker
signatures:
  - "func Email(count int)"
variantHelpers:
  - plugin#faker#email
similarHelpers:
  - plugin#faker#name
  - plugin#faker#url
position: 4
---

Creates an Observable that emits `count` random email addresses.

```go
import (
    "fmt"

    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.Email(3)

sub := obs.Subscribe(ro.NewObserver(
    func(email string) {
        fmt.Printf("Next: %s\n", email)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: john.smith@example.com
// Next: jane.doe@example.org
// Next: bob.johnson@example.net
// Completed
```
