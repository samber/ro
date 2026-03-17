---
name: Fake
slug: fake
sourceRef: plugins/faker/operator.go#L36
type: plugin
category: faker
signatures:
  - "func Fake[T any](count int)"
variantHelpers:
  - plugin#faker#fake
similarHelpers:
  - plugin#faker#name
  - plugin#faker#email
position: 0
---

Creates an Observable that emits `count` fake values of type T. Struct fields are populated using [faker struct tags](https://github.com/go-faker/faker#supported-tags).

```go
import (
    "fmt"

    rofaker "github.com/samber/ro/plugins/faker"
)

type Person struct {
    Name  string `faker:"name"`
    Email string `faker:"email"`
}

obs := rofaker.Fake[Person](3)

sub := obs.Subscribe(ro.NewObserver(
    func(p Person) {
        fmt.Printf("Next: %+v\n", p)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: {Name:John Smith Email:john.smith@example.com}
// Next: {Name:Jane Doe Email:jane.doe@example.org}
// Next: {Name:Bob Johnson Email:bob.johnson@example.net}
// Completed
```
