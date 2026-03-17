---
name: UUIDHyphenated
slug: uuidhyphenated
sourceRef: plugins/faker/operator.go#L155
type: plugin
category: faker
signatures:
  - "func UUIDHyphenated(count int)"
variantHelpers:
  - plugin#faker#uuidhyphenated
similarHelpers:
  - plugin#faker#uuiddigit
position: 8
---

Creates an Observable that emits `count` random hyphenated UUIDs (e.g., `550e8400-e29b-41d4-a716-446655440000`).

```go
import (
    "fmt"

    "github.com/samber/ro"
    rofaker "github.com/samber/ro/plugins/faker"
)

obs := rofaker.UUIDHyphenated(3)

sub := obs.Subscribe(ro.NewObserver(
    func(uuid string) {
        fmt.Printf("Next: %s\n", uuid)
    },
    func(err error) {
        fmt.Printf("Error: %v\n", err)
    },
    func() {
        fmt.Println("Completed")
    },
))
defer sub.Unsubscribe()

// Next: 550e8400-e29b-41d4-a716-446655440000
// Next: 6ba7b810-9dad-11d1-80b4-00c04fd430c8
// Next: 6ba7b811-9dad-11d1-80b4-00c04fd430c8
// Completed
```
